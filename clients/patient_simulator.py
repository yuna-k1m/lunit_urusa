from __future__ import annotations

import asyncio
from typing import Any

import httpx

from clients.common import ModelClientError


class PatientSimulatorClient:
    def __init__(self, *, base_url: str, api_key: str | None, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate_question(self, messages: list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise ModelClientError("LUNIT_FM_API_KEY is not configured")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        current_messages = messages
        timeout = httpx.Timeout(self.timeout, connect=min(15.0, self.timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(3):
                payload = {"model": self.model, "messages": current_messages, "stream": False}
                try:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions", headers=headers, json=payload
                    )
                    if response.status_code == 404 and current_messages:
                        # Lunit documents 404 as an expired/invalid continuation. Start a new
                        # simulator conversation without altering the caller's stored history.
                        current_messages = []
                    elif response.status_code in {429, 502, 503, 504} and attempt < 2:
                        pass
                    else:
                        response.raise_for_status()
                        return self._content(response.json())
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
                    if attempt == 2:
                        raise ModelClientError("could not reach Lunit patient simulator") from exc
                except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
                    raise ModelClientError("patient simulator returned an unusable response") from exc
                await asyncio.sleep(0.5 * (2**attempt))
        raise ModelClientError("patient simulator request failed")

    @staticmethod
    def _content(data: dict[str, Any]) -> str:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty completion")
        return content
