from __future__ import annotations

from chat_models.base import ChatRequest, ChatResult
from clients.lunit import LunitClient


class DirectL2Strategy:
    name = "direct_l2"

    def __init__(self, lunit: LunitClient) -> None:
        self.lunit = lunit

    async def complete(self, request: ChatRequest) -> ChatResult:
        content = await self.lunit.complete(request.messages)
        return ChatResult(content=content, model=self.name)
