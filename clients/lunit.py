from __future__ import annotations

import asyncio
from typing import Any

import httpx

from clients.common import ModelClientError


class LunitClient:
    def __init__(self, *, base_url: str, api_key: str | None, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> str:
        if not self.api_key:
            raise ModelClientError("LUNIT_FM_API_KEY is not configured")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "stream": False}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        timeout = httpx.Timeout(self.timeout, connect=min(15.0, self.timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions", headers=headers, json=payload
                    )
                    if response.status_code not in {429, 502, 503, 504} or attempt == 2:
                        response.raise_for_status()
                        return self._content(response.json())
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
                    if attempt == 2:
                        raise ModelClientError("could not reach Lunit L2") from exc
                except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
                    raise ModelClientError("Lunit L2 returned an unusable response") from exc
                await asyncio.sleep(0.5 * (2**attempt))
        raise ModelClientError("Lunit L2 request failed")

    @staticmethod
    def _content(data: dict[str, Any]) -> str:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty completion")
        return content
