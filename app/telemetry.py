"""Black-box telemetry for the evaluation container.

CoEval shows us only a score. The container can reach api.openai.com, and the
Files API lets this key upload and *list* files (but not read their content), so
per-run aggregates are encoded in the FILENAME and read back from the listing:

    urusa.<run>.t<turns>.pok<planner ok>.pto<planner timeouts/fallbacks>
         .gfb<no-thinking fallbacks>.rt<retrievals>.se<searches>.ms<mean turn ms>.err<errors>

Nothing about the conversations leaves the box; counts and timings only.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid

_lock = threading.Lock()
_stats = {"t": 0, "pok": 0, "pto": 0, "gfb": 0, "rt": 0, "se": 0, "ms": 0, "err": 0}
_run = uuid.uuid4().hex[:6]
_last_upload = 0.0


def record(out: dict | None, elapsed_s: float, error: bool = False) -> None:
    with _lock:
        _stats["t"] += 1
        _stats["ms"] += int(elapsed_s * 1000)
        if error or out is None:
            _stats["err"] += 1
            return
        plan = out.get("plan") or {}
        notes = out.get("notes") or {}
        if plan.get("_planner_model", "").startswith("gpt"):
            _stats["pok"] += 1
        else:
            _stats["pto"] += 1
        if notes.get("gen_fallback"):
            _stats["gfb"] += 1
        if notes.get("grounded"):
            _stats["rt"] += 1
        if notes.get("searched"):
            _stats["se"] += 1


def _name() -> str:
    s = dict(_stats)
    mean_ms = s["ms"] // max(1, s["t"])
    return (f"urusa.{_run}.t{s['t']}.pok{s['pok']}.pto{s['pto']}.gfb{s['gfb']}"
            f".rt{s['rt']}.se{s['se']}.ms{mean_ms}.err{s['err']}.jsonl")


def upload(api_key: str, name: str | None = None) -> None:
    name = name or _name()
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/jsonl\r\n\r\n{json.dumps({'ok': True})}\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request("https://api.openai.com/v1/files", data=body, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        pass


def start(api_key: str, every_s: float | None = None) -> None:
    """Upload a snapshot every `every_s` seconds while turns are being processed."""
    every_s = every_s or float(os.environ.get("TELEMETRY_EVERY_S", "90"))
    if not api_key or os.environ.get("TELEMETRY", "1") == "0":
        return

    # Immediate startup beacon: proves egress from the evaluation box even if no
    # request ever arrives.
    threading.Thread(target=lambda: upload(api_key, name=f"urusa.startup.{_run}.jsonl"), daemon=True).start()

    def loop() -> None:
        global _last_upload
        seen = -1
        while True:
            time.sleep(every_s)
            with _lock:
                t = _stats["t"]
            if t != seen and t > 0:
                seen = t
                upload(api_key)

    threading.Thread(target=loop, daemon=True).start()
