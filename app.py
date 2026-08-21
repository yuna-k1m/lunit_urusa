"""Minimal OpenAI-compatible conversation driver for the Lunit evaluator."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


DEFAULT_API_URL = "https://model.hackathon.lunit.io"
DEFAULT_MODEL = "Lunit/L2-preview"
DEFAULT_API_KEY_FILE = Path(__file__).with_name("submission_api_key")
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

app = FastAPI(title="Lunit Hackathon Conversation Driver", version="0.1.0")


def get_api_key() -> str | None:
    """Load the runtime credential, falling back to the submission-only key file."""
    api_key = os.getenv("LUNIT_FM_API_KEY")
    if api_key:
        return api_key
    try:
        return DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def openai_error(message: str, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    model = os.getenv("LUNIT_FM_MODEL", DEFAULT_MODEL)
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "lunit",
            }
        ],
    }


async def forward_chat(payload: dict[str, Any], api_key: str) -> httpx.Response:
    base_url = os.getenv("LUNIT_FM_API_URL", DEFAULT_API_URL).rstrip("/")
    timeout = httpx.Timeout(180.0, connect=15.0)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                if attempt == 2:
                    raise
            else:
                if response.status_code not in RETRYABLE_STATUS_CODES or attempt == 2:
                    return response
            await asyncio.sleep(0.5 * (2**attempt))

    assert response is not None
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        payload = await request.json()
    except ValueError:
        return openai_error("Request body must be valid JSON.", 400, "invalid_request_error")

    if not isinstance(payload, dict):
        return openai_error("Request body must be a JSON object.", 400, "invalid_request_error")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return openai_error(
            "'messages' must be a non-empty array.", 400, "invalid_request_error"
        )
    if payload.get("stream") is True:
        return openai_error(
            "Streaming is not supported by this submission.",
            400,
            "invalid_request_error",
        )

    api_key = get_api_key()
    if not api_key:
        return openai_error(
            "Server is missing LUNIT_FM_API_KEY.", 500, "server_error"
        )

    # The submitted model must always produce the final answer. Preserve the evaluator's
    # complete message history, but pin the upstream model to the configured L2 model.
    payload["model"] = os.getenv("LUNIT_FM_MODEL", DEFAULT_MODEL)

    try:
        upstream = await forward_chat(payload, api_key)
    except httpx.TimeoutException:
        return openai_error("Upstream model timed out.", 504, "upstream_error")
    except httpx.HTTPError:
        return openai_error("Could not reach the upstream model.", 502, "upstream_error")

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=content_type.split(";", 1)[0],
    )
