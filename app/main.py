"""OpenAI-compatible conversation driver for the Lunit hackathon.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Exposes GET /v1/models and POST /v1/chat/completions. The work is delegated to an
*engine* selected by $DRIVER_ENGINE:

    probe     answer every turn with a network/environment report
    harness   L2 plan -> generate -> assemble (app/engine.py)  [default]
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import engine, probe

MODEL_ID = os.environ.get("DRIVER_MODEL_ID", "lunit-urusa-driver")
ENGINE = os.environ.get("DRIVER_ENGINE", "harness")
_l2 = engine.L2Client()
_planner = engine.planner_from_env(_l2)

app = FastAPI(title="lunit-urusa driver")

# Probe once at startup in the background so the first request is fast, then
# refresh per request so we also see what the network looks like under load.
_startup_probe: dict[str, Any] = {}


def _bg_probe() -> None:
    _startup_probe.update(probe.run_probe())


@app.on_event("startup")
def _on_startup() -> None:
    if ENGINE == "probe":
        threading.Thread(target=_bg_probe, daemon=True).start()


class Message(BaseModel):
    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    # anything else the evaluator sends, we want to see it
    model_config = {"extra": "allow"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content-part arrays
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "engine": ENGINE}


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "lunit-urusa"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, raw: Request) -> JSONResponse:
    t0 = time.time()
    if ENGINE == "probe":
        answer = _probe_answer(req, raw)
    else:
        msgs = [{"role": m.role, "content": _text(m.content)} for m in req.messages]
        out = await run_in_threadpool(
            engine.answer, _l2, msgs, temperature=req.temperature or 0.3,
            max_tokens=req.max_tokens or 2048, planner=_planner,
        )
        answer = out["answer"]

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(t0),
        "model": req.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
    return JSONResponse(body)


def _probe_answer(req: ChatRequest, raw: Request) -> str:
    extra = {k: v for k, v in (req.model_extra or {}).items()}
    request_info = {
        "model": req.model,
        "n_messages": len(req.messages),
        "roles": [m.role for m in req.messages],
        "message_chars": [len(_text(m.content)) for m in req.messages],
        "last_user_head": _text(req.messages[-1].content)[:120] if req.messages else None,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
        "extra_fields": extra,
        "headers": {
            k: ("<redacted>" if k.lower() == "authorization" else v)
            for k, v in raw.headers.items()
        },
        "client": raw.client.host if raw.client else None,
    }
    live = probe.run_probe()
    report = probe.format_report(live, request_info)
    if _startup_probe:
        diff = {
            k: (_startup_probe["https"].get(k), v)
            for k, v in live["https"].items()
            if _startup_probe["https"].get(k, "")[:8] != v[:8]
        }
        report += "\n\nstartup-vs-now https diffs: " + (str(diff) if diff else "none")
    return report
