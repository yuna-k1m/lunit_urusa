"""Self-contained OpenAI-compatible smoke-test service with no network calls."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


MODEL_ID = "urusa-smoke-test"
app = FastAPI(title="Urusa CoEval Smoke Test", version="0.1.0")


def openai_error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": "messages",
                "code": None,
            }
        },
    )


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "urusa"}],
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
        return openai_error("'messages' must be a non-empty array.")
    if payload.get("stream") is True:
        return openai_error("Streaming is not supported by this smoke-test service.")

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Submission container is running.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
