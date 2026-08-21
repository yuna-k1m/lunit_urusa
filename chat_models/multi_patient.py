from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from chat_models.base import ChatRequest, ChatResult
from clients.lunit import LunitClient
from clients.openai_sol import SolClient
from pipeline.conversation import serialize_conversation
from pipeline.profiles import PatientProfile, parse_profiles, profile_schema
from prompts.aggregator import AGGREGATOR_INSTRUCTIONS
from prompts.profile_generator import PROFILE_GENERATOR_INSTRUCTIONS, profile_generator_input


class MultiPatientStrategy:
    name = "multi_patient"

    def __init__(
        self, *, lunit: LunitClient, sol: SolClient, profile_count: int,
        max_parallel: int, minimum_successes: int, pipeline_timeout: float,
    ) -> None:
        self.lunit = lunit
        self.sol = sol
        self.profile_count = profile_count
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.minimum_successes = minimum_successes
        self.pipeline_timeout = pipeline_timeout

    async def complete(self, request: ChatRequest) -> ChatResult:
        try:
            return await asyncio.wait_for(self._run(request), timeout=self.pipeline_timeout)
        except Exception:
            # The direct path is the safest degradation for malformed model output,
            # missing Sol credentials, or a pipeline-wide timeout.
            content = await self.lunit.complete(request.messages)
            return ChatResult(content=content, model=self.name, metadata={"fallback": "direct_l2"})

    async def _run(self, request: ChatRequest) -> ChatResult:
        conversation = serialize_conversation(request.messages)
        profiles = await self._generate_profiles(conversation)
        tasks = [asyncio.create_task(self._ask_l2(profile)) for profile in profiles]
        done, pending = await asyncio.wait(tasks, timeout=max(0.1, self.pipeline_timeout * 0.6))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results: list[Any] = [
            task.result() if task in done and not task.cancelled() and task.exception() is None
            else RuntimeError("profile call failed or timed out")
            for task in tasks
        ]
        answers = [
            {"profile": asdict(profile), "answer": result}
            for profile, result in zip(profiles, results)
            if isinstance(result, str)
        ]
        if len(answers) < self.minimum_successes:
            raise RuntimeError("too few L2 profile calls succeeded")

        aggregation_input = json.dumps(
            {"original_conversation": request.messages, "candidate_analyses": answers},
            ensure_ascii=False,
        )
        content = await self.sol.generate_text(
            instructions=AGGREGATOR_INSTRUCTIONS, input_text=aggregation_input
        )
        return ChatResult(
            content=content,
            model=self.name,
            metadata={"profiles_requested": self.profile_count, "profiles_succeeded": len(answers), "finalizer": "sol"},
        )

    async def _generate_profiles(self, conversation: str) -> list[PatientProfile]:
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                generated = await self.sol.generate_json(
                    instructions=PROFILE_GENERATOR_INSTRUCTIONS,
                    input_text=profile_generator_input(conversation, self.profile_count),
                    schema_name="patient_profiles",
                    schema=profile_schema(self.profile_count),
                )
                return parse_profiles(generated, self.profile_count)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def _ask_l2(self, profile: PatientProfile) -> str:
        prompt = (
            "Answer this self-contained medical query. Treat its stated facts as authoritative, "
            "distinguish facts from assumptions, and respond in the user's language.\n\n"
            f"Analysis perspective: {profile.perspective}\n"
            f"Query: {profile.modified_query}"
        )
        async with self.semaphore:
            return await self.lunit.complete([{"role": "user", "content": prompt}])
