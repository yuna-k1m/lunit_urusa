"""Verified specifics via a web-searching frontier model (OpenAI Responses API).

HealthBench rubrics reward exact elements a physician would state — "AAP recommends
X for children >= 6 months", "seek evaluation within 48 hours", "max 3 g/day" —
and they are hard to predict from the question alone (h12: sol's own checklist
overlapped only 38% of missed items). Fetching the source language narrows that
gap. This module asks a search-capable model for 3-6 citable specifics; the
engine then tells L2 to state them explicitly. L2 still writes every word.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

SEARCH_SYSTEM = (
    "You find the exact, citable specifics a careful physician would state when answering the user's "
    "health question. Output ONLY JSON: {\"specifics\": [{\"fact\": \"<one sentence containing the exact "
    "number, timeframe, threshold, dose, or named recommendation as the source states it>\", \"source\": "
    "\"<organization or guideline, year>\"}]} with 3-6 items. Prefer CDC, AAP, ACOG, AHA/ACC, USPSTF, WHO, "
    "NICE, major specialty societies, FDA/DailyMed labels, and for non-US users the relevant national body. "
    "Include: the decisive recommendation with its timeframe or threshold; the red flags that mandate urgent "
    "care; dose limits if a medication is involved; what the guideline says NOT to do. No generic advice; "
    "specifics only. If the question is purely about the user's own situation with no citable standard, "
    "return {\"specifics\": []}."
)


def search_specifics(api_key: str, question: str, *, model: str | None = None,
                     timeout: float = 40.0) -> tuple[list[dict], dict]:
    """Returns (specifics, meta). Best effort: any failure yields ([], meta)."""
    model = model or os.environ.get("SEARCH_MODEL", "gpt-5.6-sol")
    body = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "reasoning": {"effort": os.environ.get("SEARCH_REASONING_EFFORT", "low")},
        "input": [{"role": "system", "content": SEARCH_SYSTEM},
                  {"role": "user", "content": question[:4000]}],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as e:
        return [], {"error": str(e)[:160], "sec": round(time.time() - t0, 1)}
    text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []) if c.get("type") == "output_text")
    try:
        items = json.loads(text).get("specifics", [])
    except (json.JSONDecodeError, AttributeError):
        items = []
    out = [{"fact": str(i.get("fact", "")).strip(), "source": str(i.get("source", "")).strip()}
           for i in items if isinstance(i, dict) and i.get("fact")][:6]
    return out, {"sec": round(time.time() - t0, 1),
                 "searches": sum(1 for o in d.get("output", []) if o.get("type") == "web_search_call"),
                 "tokens": (d.get("usage") or {}).get("total_tokens")}


def format_specifics(items: list[dict]) -> str:
    return "\n".join(f"  - {i['fact']} ({i['source']})" if i["source"] else f"  - {i['fact']}" for i in items)
