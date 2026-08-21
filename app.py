"""Minimal OpenAI-compatible conversation driver for Lunit FM L2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DEFAULT_API_URL = "https://model.hackathon.lunit.io"
DEFAULT_MODEL = "Lunit/L2-preview"
REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_API_KEY_FILE = Path(__file__).with_name("submission_api_key")

app = FastAPI(title="Urusa Conversation Driver", version="0.2.0")


def get_api_key() -> str | None:
    """Prefer an injected key, then use the key bundled for CoEval."""
    api_key = os.getenv("LUNIT_FM_API_KEY")
    if api_key:
        return api_key
    try:
        return DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def settings() -> tuple[str, str, str | None]:
    return (
        os.getenv("LUNIT_FM_API_URL", DEFAULT_API_URL).rstrip("/"),
        os.getenv("LUNIT_FM_MODEL", DEFAULT_MODEL),
        get_api_key(),
    )


def openai_error(
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": param, "code": None}},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    _, model, _ = settings()
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "created": 0, "owned_by": "lunit"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError:
        return openai_error("Request body must be valid JSON.")

    if not isinstance(payload, dict):
        return openai_error("Request body must be a JSON object.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return openai_error("'messages' must be a non-empty array.", param="messages")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            return openai_error(
                f"messages[{index}] must contain a string 'role'.", param="messages"
            )
        if "content" not in message:
            return openai_error(f"messages[{index}] must contain 'content'.", param="messages")
    if payload.get("stream") is True:
        return openai_error("Streaming is not supported.", param="stream")

    api_url, model, api_key = settings()
    if not api_key:
        return openai_error(
            "Server is missing LUNIT_FM_API_KEY.",
            status_code=503,
            error_type="server_error",
        )

    # Retain standard generation parameters and the complete conversation history.
    upstream_payload = dict(payload)
    upstream_payload["model"] = model
    upstream_payload["messages"] = messages

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{api_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=upstream_payload,
            )
    except httpx.TimeoutException:
        return openai_error(
            "The upstream model timed out.", status_code=504, error_type="upstream_timeout"
        )
    except httpx.RequestError:
        return openai_error(
            "The upstream model is unavailable.", status_code=502, error_type="upstream_error"
        )

    try:
        body = response.json()
    except ValueError:
        return openai_error(
            "The upstream model returned an invalid response.",
            status_code=502,
            error_type="upstream_error",
        )
    return JSONResponse(status_code=response.status_code, content=body)
