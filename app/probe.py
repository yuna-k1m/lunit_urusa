"""Network/environment probe for the evaluation box.

The eval container is a black box to us: we can't log in, and the only channel
back is whatever the dashboard shows for a trial (response text, score, errors).
So the probe server answers every chat completion with a report of what the
container can actually see: DNS, outbound HTTPS to the Lunit endpoints and to
third-party model APIs, which env vars are injected, and the shape of the
request the evaluator sent.

Stdlib only so the report works even if a pip install went wrong.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import ssl
import time
import urllib.error
import urllib.request

# (label, url). 401/403 counts as "reachable" -- we only care about the network path.
TARGETS = [
    ("lunit-model", "https://model.hackathon.lunit.io/v1/models"),
    ("lunit-mcp", "https://mcp.hackathon.lunit.io/mcp"),
    ("lunit-patient", "https://patient.hackathon.lunit.io/v1/models"),
    ("openai", "https://api.openai.com/v1/models"),
    ("anthropic", "https://api.anthropic.com/v1/models"),
    ("google-genai", "https://generativelanguage.googleapis.com/"),
    ("openrouter", "https://openrouter.ai/api/v1/models"),
    ("pypi", "https://pypi.org/simple/"),
    ("github", "https://github.com/"),
    ("cloudflare-dns", "https://1.1.1.1/"),
]

# Env var *names* only. Values are never reported (the key would land in a dashboard).
ENV_PREFIXES = ("LUNIT", "OPENAI", "ANTHROPIC", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "no_proxy", "MODEL", "API", "EVAL", "HACKATHON")


def _resolve(host: str) -> str:
    try:
        t = time.time()
        ip = socket.gethostbyname(host)
        return f"{ip} ({(time.time() - t) * 1000:.0f}ms)"
    except OSError as e:
        return f"FAIL {e}"


def _get(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "lunit-probe/1"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"HTTP {r.status} ({(time.time() - t) * 1000:.0f}ms)"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} ({(time.time() - t) * 1000:.0f}ms)"
    except urllib.error.URLError as e:
        return f"FAIL {e.reason} ({(time.time() - t) * 1000:.0f}ms)"
    except (OSError, ssl.SSLError) as e:
        return f"FAIL {e} ({(time.time() - t) * 1000:.0f}ms)"


def run_probe() -> dict:
    out: dict = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_present": sorted(
            k for k in os.environ if any(k.startswith(p) for p in ENV_PREFIXES)
        ),
        "dns": {},
        "https": {},
    }
    for label, url in TARGETS:
        host = url.split("/")[2]
        out["dns"][host] = _resolve(host)
        out["https"][label] = _get(url)
    return out


def format_report(probe: dict, request_info: dict) -> str:
    lines = ["[LUNIT PROBE REPORT]", ""]
    lines.append(f"host={probe['hostname']} python={probe['python']} utc={probe['utc']}")
    lines.append(f"platform={probe['platform']}")
    lines.append("env_present=" + (", ".join(probe["env_present"]) or "(none)"))
    lines.append("")
    lines.append("HTTPS reachability:")
    for k, v in probe["https"].items():
        lines.append(f"  {k:<16} {v}")
    lines.append("")
    lines.append("DNS:")
    for k, v in probe["dns"].items():
        lines.append(f"  {k:<44} {v}")
    lines.append("")
    lines.append("Request as received:")
    lines.append(json.dumps(request_info, ensure_ascii=False, indent=2))
    return "\n".join(lines)
