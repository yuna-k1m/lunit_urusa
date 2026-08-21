from __future__ import annotations

import asyncio
import json

from chat_models.base import ChatRequest, ChatResult
from clients.lunit import LunitClient
from clients.openai_sol import SolClient
from clients.patient_simulator import PatientSimulatorClient
from prompts.patient_sim import PATIENT_FINALIZER_INSTRUCTIONS, PATIENT_PREP_INSTRUCTIONS


class PatientSimStrategy:
    name = "patient-sim"

    def __init__(
        self, *, patient_simulator: PatientSimulatorClient, lunit: LunitClient,
        sol: SolClient, pipeline_timeout: float,
    ) -> None:
        self.patient_simulator = patient_simulator
        self.lunit = lunit
        self.sol = sol
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
        simulated_question = await self.patient_simulator.generate_question(request.messages)
        prep_input = json.dumps(
            {"original_conversation": request.messages, "simulated_patient_question": simulated_question},
            ensure_ascii=False,
        )
        doctor_prep = await self.sol.generate_text(
            instructions=PATIENT_PREP_INSTRUCTIONS, input_text=prep_input
        )
        diagnosis_prompt = (
            "Provide a diagnostic assessment for the doctor based on the intake material below. "
            "Distinguish known facts from simulated or missing details, give a prioritized differential, "
            "explain supporting and opposing findings, identify red flags, and suggest appropriate next "
            "steps. Do not present an uncertain diagnosis as confirmed.\n\n"
            f"{doctor_prep}"
        )
        diagnosis = await self.lunit.complete(
            [{"role": "user", "content": diagnosis_prompt}], max_tokens=max_tokens
        )
        final_input = json.dumps(
            {
                "original_conversation": request.messages,
                "simulated_patient_question": simulated_question,
                "doctor_preparation": doctor_prep,
                "l2_diagnostic_assessment": diagnosis,
            },
            ensure_ascii=False,
        )
        content = await self.sol.generate_text(
            instructions=PATIENT_FINALIZER_INSTRUCTIONS,
            input_text=final_input,
            max_output_tokens=max_tokens,
        )
        return ChatResult(
            content=content,
            model=self.name,
            metadata={"simulator_questions": 1, "diagnostician": "l2", "finalizer": "sol"},
        )

    @staticmethod
    def _max_tokens(request: ChatRequest) -> int:
        value = request.original_payload.get("max_tokens", 4096)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 4096
