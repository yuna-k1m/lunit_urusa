"""Adapter for siusiubeom's h4 planner -> L2 writer -> assembler harness."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from app import engine
from chat_models.base import ChatRequest, ChatResult


class SiusiubeomH4Strategy:
    name = "siusiubeom_h4"

    def __init__(
        self,
        client: engine.L2Client | None = None,
        planner: engine.L2Client | None | object = ...,
        local_first: bool = False,
    ) -> None:
        self.client = client or engine.L2Client()
        self.planner = (
            engine.planner_from_env(self.client)
            if planner is ...
            else planner
        )
        self.local_first = local_first

    async def complete(self, request: ChatRequest) -> ChatResult:
        temperature = float(os.environ.get("GEN_TEMPERATURE", self._number(request.original_payload.get("temperature"), 0.3)))
        # The evaluator's max_tokens is not passed through: L2 spends thousands of
        # tokens on a hidden reasoning channel first, and a 2048 cap truncates the
        # visible answer (engine.GEN_MAX_TOKENS is the tuned budget).
        output = await asyncio.to_thread(
            engine.answer,
            self.client,
            request.messages,
            temperature=temperature,
            max_tokens=None,
            planner=self.planner,
            local_first=self.local_first,
        )
        metadata: dict[str, Any] = {
            key: output.get(key)
            for key in ("plan", "notes", "review", "retrieval", "timings")
        }
        return ChatResult(content=output["answer"], model=self.name, metadata=metadata)

    @staticmethod
    def _number(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
