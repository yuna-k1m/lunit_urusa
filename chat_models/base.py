from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, Any]]
    original_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatStrategy(Protocol):
    name: str

    async def complete(self, request: ChatRequest) -> ChatResult: ...
