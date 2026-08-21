from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from chat_models.base import ChatRequest, ChatResult
from clients.lunit import LunitClient
from clients.openai_sol import SolClient
from pipeline.conversation import serialize_conversation
from pipeline.demographics import DemographicProfile, predefined_demographics
from prompts.demographic_aggregator import DEMOGRAPHIC_AGGREGATOR_INSTRUCTIONS


class DemographicsStrategy:
    name = "demographics"

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
        max_tokens = self._max_tokens(request)
        try:
            return await asyncio.wait_for(
                self._run(request, max_tokens=max_tokens), timeout=self.pipeline_timeout
            )
        except Exception:
            content = await self.lunit.complete(request.messages, max_tokens=max_tokens)
            return ChatResult(content=content, model=self.name, metadata={"fallback": "direct_l2"})

    async def _run(self, request: ChatRequest, *, max_tokens: int) -> ChatResult:
        conversation = serialize_conversation(request.messages)
        profiles = predefined_demographics(conversation, self.profile_count)
        tasks = [
            asyncio.create_task(self._ask_l2(profile, max_tokens=max_tokens))
            for profile in profiles
        ]
        done, pending = await asyncio.wait(tasks, timeout=max(0.1, self.pipeline_timeout * 0.6))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results: list[Any] = [
            task.result() if task in done and not task.cancelled() and task.exception() is None
            else RuntimeError("demographic call failed or timed out")
            for task in tasks
        ]
        answers = [
            {"profile": asdict(profile), "answer": result}
            for profile, result in zip(profiles, results)
            if isinstance(result, str)
        ]
        if len(answers) < self.minimum_successes:
            raise RuntimeError("too few L2 demographic calls succeeded")

        aggregation_input = json.dumps(
            {"raw_input": request.messages, "l2_answers": answers},
            ensure_ascii=False,
        )
        content = await self.sol.generate_text(
            instructions=DEMOGRAPHIC_AGGREGATOR_INSTRUCTIONS,
            input_text=aggregation_input,
            max_output_tokens=max_tokens,
        )
        return ChatResult(
            content=content,
            model=self.name,
            metadata={
                "profiles_requested": self.profile_count,
                "profiles_succeeded": len(answers),
                "finalizer": "sol",
                "profile_type": "demographic_sensitivity",
            },
        )

    async def _ask_l2(self, profile: DemographicProfile, *, max_tokens: int) -> str:
        known = "\n".join(f"- {fact}" for fact in profile.preserved_facts) or "- None stated"
        hypothetical = (
            "\n".join(f"- {fact}" for fact in profile.hypothetical_facts)
            or "- None; all demographic attributes are stated facts or left unspecified"
        )
        prompt = (
            "Answer this self-contained medical query in the user's language. The original "
            "conversation is authoritative and overrides this preset persona wherever they "
            "conflict. In that case, ignore the conflicting persona attribute. Do not imply that "
            "the original user has any hypothetical attribute. Distinguish universal guidance "
            "from guidance that depends on the sensitivity scenario.\n\n"
            f"Authoritative preserved facts:\n{known}\n\n"
            f"Hypothetical demographic sensitivity assumptions:\n{hypothetical}\n\n"
            f"Age group: {profile.age_group or 'unspecified'}\n"
            f"Sex at birth: {profile.sex_at_birth or 'unspecified'}\n"
            f"Gender: {profile.gender or 'unspecified'}\n"
            f"Analysis perspective: {profile.perspective}\n"
            f"Focus areas: {', '.join(profile.focus_areas)}\n"
            f"Query: {profile.modified_query}"
        )
        async with self.semaphore:
            return await self.lunit.complete(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens
            )

    @staticmethod
    def _max_tokens(request: ChatRequest) -> int:
        value = request.original_payload.get("max_tokens", 4096)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 4096
