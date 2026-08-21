from __future__ import annotations

import json
from typing import Any

import httpx

from clients.common import ModelClientError


class SolClient:
    def __init__(self, *, base_url: str, api_key: str | None, model: str, reasoning_effort: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    async def generate_text(
        self, *, instructions: str, input_text: str, max_output_tokens: int | None = None
    ) -> str:
        return await self._request(
            instructions=instructions, input_text=input_text, max_output_tokens=max_output_tokens
        )

    async def generate_json(
        self, *, instructions: str, input_text: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        text = await self._request(
            instructions=instructions,
            input_text=input_text,
            text_format={"type": "json_schema", "name": schema_name, "strict": True, "schema": schema},
        )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelClientError("GPT Sol returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ModelClientError("GPT Sol structured output was not an object")
        return value

    async def _request(
        self, *, instructions: str, input_text: str, text_format: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise ModelClientError("OPENAI_API_KEY is not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
        }
        if text_format:
            payload["text"] = {"format": text_format}
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=15.0)) as client:
                response = await client.post(f"{self.base_url}/responses", headers=headers, json=payload)
                response.raise_for_status()
                return self._output_text(response.json())
        except httpx.HTTPError as exc:
            raise ModelClientError("GPT Sol request failed") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelClientError("GPT Sol returned an unusable response") from exc

    @staticmethod
    def _output_text(data: dict[str, Any]) -> str:
        direct = data.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return content["text"]
        raise ValueError("response has no output text")
