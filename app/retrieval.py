"""L2 retrieval stage over the Lunit MCP tools.

    evidence = run_retrieval(l2, "recommended blood pressure target in adults with CKD")

The retrieval stage is its own L2 conversation: a retrieval-only system prompt,
the MCP tools plus a locally defined `finalize_retrieval`, and a hard cap on tool
calls. Every tool result is scanned for citable items (dicts carrying a
`cite_uid`) and cached, so the `cite_uid`s the model reports at the end can be
resolved back to content for the generation stage. The stage never answers the
question.

MCP here is Streamable HTTP, stateless (no session id), JSON-RPC 2.0. Stdlib only.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.engine import L2Client, resolve_key
from clients.health_db import HealthDBTool, TOOL_NAME as HEALTH_DB_TOOL_NAME

PROMPTS = Path(__file__).resolve().parent / "prompts"
RETRIEVAL_SYSTEM = (PROMPTS / "retrieval.md").read_text(encoding="utf-8").strip()

MCP_URL = os.environ.get("LUNIT_MCP_URL", "https://mcp.hackathon.lunit.io/mcp")
MAX_TOOL_CALLS = int(os.environ.get("RETRIEVAL_MAX_CALLS", "4"))
MAX_SECONDS = float(os.environ.get("RETRIEVAL_MAX_SECONDS", "25"))
TOOL_RESULT_CHARS = 6000      # what the model sees per tool result
ITEM_CONTENT_CHARS = 3500     # what the generation stage sees per cited item

FINALIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize_retrieval",
        "description": (
            "Submit your final citation selection and end the retrieval phase. Call this only: "
            "once you have gathered enough evidence to answer the query; or the query does not "
            "need any retrieval; or you exhausted the tool call budget and must end the retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["sufficient", "partial", "no_evidence"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cite_uid": {"type": "string"},
                            "relevance_score": {"type": "number"},
                        },
                        "required": ["cite_uid", "relevance_score"],
                    },
                },
                "note": {"type": "string"},
            },
            "required": ["status", "items"],
        },
    },
}


# -------------------------------------------------------------------------- MCP


class MCPClient:
    def __init__(self, url: str = MCP_URL, key: str | None = None, timeout: float = 60.0) -> None:
        self.url = url
        self.key = key or resolve_key("LUNIT_FM_API_KEY", "LUNIT_KEY")
        self.timeout = timeout
        self._tools: list[dict] | None = None
        self._lock = threading.Lock()
        self.health_db = HealthDBTool()

    def _rpc(self, method: str, params: dict) -> dict:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {self.key}",
            },
        )
        raw = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read().decode()
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except OSError:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        # Streamable HTTP may answer as SSE ("data: {...}") or as plain JSON.
        events = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:")]
        msg = events[-1] if events else json.loads(raw)
        if "error" in msg:
            raise RuntimeError(f"MCP {method}: {msg['error']}")
        return msg.get("result", {})

    def tools(self) -> list[dict]:
        with self._lock:
            if self._tools is None:
                try:
                    self._tools = self._rpc("tools/list", {}).get("tools", [])
                except Exception:
                    if not self.health_db.available:
                        raise
                    self._tools = []
            tools = list(self._tools)
            if self.health_db.available:
                tools.append(self.health_db_tool())
            return tools

    def health_db_tool(self) -> dict:
        from clients.health_db import TOOL

        return TOOL

    def call(self, name: str, arguments: dict) -> str:
        if name == HEALTH_DB_TOOL_NAME:
            return self.health_db.call(arguments)
        res = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts = res.get("content") or []
        text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if res.get("isError"):
            return f"TOOL ERROR: {text[:500]}"
        return text


# Tools hidden from the model. Section trees and document lists carry no cite_uid,
# and L2 loops on them (3-4 of 5 calls in tests); `index_get_relevant_nodes` plus the
# harness auto-read of the top hit's pages covers the same ground in one step.
HIDDEN_TOOLS = set(filter(None, os.environ.get(
    "RETRIEVAL_HIDDEN_TOOLS", "index_get_document_structure,index_list_documents").split(",")))


def to_openai_tools(mcp_tools: list[dict]) -> list[dict]:
    out = []
    for t in mcp_tools:
        if t["name"] in HIDDEN_TOOLS:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": (t.get("description") or "")[:1500],
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out + [FINALIZE_TOOL]


# -------------------------------------------------------------------- citations


def _walk_citables(obj: Any, sink: dict[str, dict], tool: str) -> None:
    if isinstance(obj, dict):
        uid = obj.get("cite_uid")
        if isinstance(uid, str) and uid and uid not in sink:
            item = dict(obj)
            item["_tool"] = tool
            sink[uid] = item
        for v in obj.values():
            _walk_citables(v, sink, tool)
    elif isinstance(obj, list):
        for v in obj:
            _walk_citables(v, sink, tool)


def with_hints(tool: str, args: dict, text: str) -> str:
    """Append the exact follow-up call to node lists, so the model reads pages
    (which carry cite_uids) instead of browsing document structures."""
    if tool != "index_get_relevant_nodes":
        return text
    try:
        nodes = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    if not isinstance(nodes, list):
        return text
    hints = []
    for n in nodes[:3]:
        if isinstance(n, dict) and n.get("doc_id") and isinstance(n.get("range"), list) and len(n["range"]) == 2:
            a, b = n["range"]
            b = min(int(b), int(a) + 4)
            hints.append(
                f"index_get_page_content(corpus_tag=\"{args.get('corpus_tag', 'guideline')}\", "
                f"doc_id=\"{n['doc_id']}\", start_page={a}, end_page={b})  # {str(n.get('doc_title', ''))[:60]}"
            )
    if not hints:
        return text
    return text + "\n\nTo cite any of these, read the pages (only page content has a cite_uid):\n" + "\n".join(hints)


def auto_read_top(mcp: "MCPClient", args: dict, nodes_text: str) -> str:
    try:
        nodes = json.loads(nodes_text.split("\n\nTo cite any of these")[0])
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(nodes, list):
        return ""
    for n in nodes[:1]:
        if isinstance(n, dict) and n.get("doc_id") and isinstance(n.get("range"), list) and len(n["range"]) == 2:
            a, b = int(n["range"][0]), int(n["range"][1])
            try:
                return mcp.call("index_get_page_content", {
                    "corpus_tag": args.get("corpus_tag", "guideline"), "doc_id": n["doc_id"],
                    "start_page": a, "end_page": min(b, a + 2)})
            except Exception:
                return ""
    return ""


def harvest(text: str, sink: dict[str, dict], tool: str) -> None:
    """Cache every citable item found in a tool result (JSON or JSON-ish text)."""
    try:
        _walk_citables(json.loads(text), sink, tool)
    except (json.JSONDecodeError, ValueError):
        return


def _title_of(item: dict) -> str:
    for k in ("title", "doc_title", "drug_name", "name", "law_name", "source_id"):
        if item.get(k):
            t = str(item[k])
            return t + (f" — {item['section']}" if item.get("section") else "")
    return item.get("source_type", "source")


def _content_of(item: dict) -> str:
    for k in ("content", "text", "page_content", "summary", "body"):
        if item.get(k):
            return str(item[k])
    if isinstance(item.get("pages"), list):  # index_get_page_content
        return "\n\n".join(
            f"(p.{p.get('page')}) {p.get('text', '')}" for p in item["pages"] if isinstance(p, dict)
        )
    # fall back to the remaining scalar fields
    skip = {"cite_uid", "url", "source_type", "source_id", "_tool", "layer", "tool_result_type"}
    return "\n".join(f"{k}: {v}" for k, v in item.items()
                     if k not in skip and isinstance(v, (str, int, float)))


def format_evidence(result: dict) -> str:
    """Render the finalized selection the way the generation stage expects it."""
    lines = [f"status: {result['status']}"]
    if result.get("note"):
        lines.append(f"note: {result['note']}")
    for i, it in enumerate(result["items"], 1):
        lines.append("")
        lines.append(f"[{i}]")
        lines.append(f"source_type: {it.get('source_type', it.get('_tool', 'unknown'))}")
        if it.get("url"):
            lines.append(f"url: {it['url']}")
        lines.append(f"title: {_title_of(it)}")
        lines.append("content: " + _content_of(it)[:ITEM_CONTENT_CHARS])
    return "\n".join(lines)


# ---------------------------------------------------------------------- stage


SELECTOR_SYSTEM = (
    "You select evidence. Given a query and numbered candidate items (title + snippet), output ONLY a JSON "
    'object {"selected": [indices of items that directly help answer the query, most relevant first, at most 4], '
    '"status": "sufficient" | "partial" | "no_evidence", "note": "one sentence on what the evidence covers and lacks"}. '
    "Select nothing that is off-topic; an empty list is a valid answer."
)


def select_with_model(selector: L2Client, query: str, cache: dict[str, dict]) -> dict | None:
    """Ask a (strong) model which cached items are relevant. Returns {'items','status','note'} or None."""
    uids = list(cache)[:40]
    lines = []
    for i, uid in enumerate(uids, 1):
        it = cache[uid]
        lines.append(f"[{i}] {_title_of(it)[:120]} :: {_content_of(it)[:300].replace(chr(10), ' ')}")
    user = "Query: " + query + "\n\nCandidates:\n" + "\n".join(lines) + "\n\nOutput the JSON."
    try:
        # L2 as selector (no external model): thinking off keeps it to a few seconds.
        text = selector.chat_message([{"role": "system", "content": SELECTOR_SYSTEM}, {"role": "user", "content": user}],
                                     temperature=0.0, max_tokens=400, response_format={"type": "json_object"},
                                     thinking=False if not selector.reasoning_style else None,
                                     retries=1, timeout=25.0).get("content") or ""
        obj = json.loads(text)
    except Exception:
        return None
    def _flat(v):
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, list):
                yield from _flat(x)
            else:
                yield x
    idx = []
    for i in _flat(obj.get("selected", [])):
        if str(i).isdigit() and 1 <= int(i) <= len(uids) and int(i) not in idx:
            idx.append(int(i))
    return {"items": [cache[uids[i - 1]] for i in idx][:4],
            "status": obj.get("status", "partial" if idx else "no_evidence"),
            "note": str(obj.get("note", ""))[:300]}


_TITLES_FILE = PROMPTS / "guideline_titles.json"
KNOWN_TITLES: list[str] = json.loads(_TITLES_FILE.read_text(encoding="utf-8")) if _TITLES_FILE.exists() else []


def run_retrieval(l2: L2Client, query: str, *, mcp: MCPClient | None = None,
                  max_calls: int = MAX_TOOL_CALLS, temperature: float = 0.0,
                  selector: L2Client | None = None, hints: dict | None = None,
                  local_limit: int | None = None, local_first: bool = False) -> dict:
    """Run the retrieval stage for one self-contained query.

    Returns {"status", "items": [resolved citable items], "note", "trace": [...],
             "tool_calls": n, "elapsed": s}."""
    mcp = mcp or MCPClient()
    t0 = time.time()
    cache: dict[str, dict] = {}
    trace: list[dict] = []
    messages: list[dict] = [
        {"role": "system", "content": RETRIEVAL_SYSTEM},
        {"role": "user", "content": query},
    ]
    final: dict | None = None
    calls = 0
    seeded = False
    local_hits: list[dict] = []
    local_mcp_seeded = False

    # L2-plus invariant: local catalog lookup happens before every remote MCP
    # tool call. Catalog node ids then route directly to relevant guideline
    # sections, avoiding a broad remote document search.
    health_db = getattr(mcp, "health_db", None)
    if local_first and health_db is not None and health_db.available:
        limit = local_limit or int(os.environ.get("L2_PLUS_LOCAL_LIMIT", "5"))
        try:
            local_hits = health_db.search(query, limit=limit)
        except Exception as exc:
            trace.append({"tool": "local:health_db_search", "error": str(exc)[:200], "sec": 0})
        else:
            trace.append({"tool": "local:health_db_search", "query": query,
                          "hits": len(local_hits), "sec": round(time.time() - t0, 3)})
            from app.seed import seed_guideline_node
            for hit in local_hits[:2]:
                try:
                    catalog = json.loads(hit.get("content") or "{}")
                except (json.JSONDecodeError, TypeError):
                    catalog = {}
                node_id = catalog.get("node_id") if isinstance(catalog, dict) else None
                if node_id and "index_list_documents" in str(hit.get("source", "")):
                    seed_guideline_node(mcp, str(node_id), query, cache, trace)
            local_mcp_seeded = bool(cache)
            # If MCP did not yield citable pages, retain catalog summaries as a
            # transparent fallback rather than pretending no local evidence exists.
            if not cache:
                for hit in local_hits:
                    cache[hit["cite_uid"]] = hit

    if local_mcp_seeded:
        seeded = True
        final = {"status": "partial", "items": [], "note": "local DB routed MCP evidence"}
        max_calls = 0

    # Deterministic seeds from the planner's structured hints (see app/seed.py).
    # If they yield citable items, skip the L2 loop: faster and more reliable.
    from app import seed as _seed
    h = _seed.normalize_hints(hints)
    if _seed.has_any(h):
        _seed.run_seeds(mcp, h, cache, trace, KNOWN_TITLES)
        seeded = True
        if cache:
            final = {"status": "partial", "items": [], "note": "seeded"}
            max_calls = 0

    # Fetching the remote tool list is intentionally after the local lookup and
    # deterministic local-hit seeds.
    if max_calls == 0:
        tools = [FINALIZE_TOOL]
    else:
        remote_tools = mcp.tools()
        if not local_first:
            remote_tools = [tool for tool in remote_tools if tool.get("name") != HEALTH_DB_TOOL_NAME]
        tools = to_openai_tools(remote_tools)

    for round_ in range(max_calls + 2):
        if final is not None and max_calls == 0:
            break
        out_of_budget = (max_calls - calls) <= 0 or (time.time() - t0) > MAX_SECONDS
        if out_of_budget:
            # Budget gone: force finalize_retrieval as the only possible action.
            messages.append({"role": "user", "content":
                             "Tool budget exhausted. Call finalize_retrieval now with the relevant cite_uids "
                             "(status 'partial' if evidence is incomplete, 'no_evidence' if none)."})
            msg = l2.chat_message(messages, tools=[FINALIZE_TOOL], temperature=temperature, max_tokens=2000,
                                  tool_choice={"type": "function", "function": {"name": "finalize_retrieval"}})
        else:
            msg = l2.chat_message(messages, tools=tools, temperature=temperature, max_tokens=3000)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # the model spoke instead of calling a tool: nudge once, then stop
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            if out_of_budget or round_ >= max_calls:
                break
            messages.append({"role": "user", "content":
                             "Do not answer. Either call another tool or call finalize_retrieval."})
            continue
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "finalize_retrieval":
                final = {"status": args.get("status", "partial"), "items": args.get("items") or [],
                         "note": args.get("note", "")}
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "ok"})
                continue
            calls += 1
            t = time.time()
            try:
                out = mcp.call(name, args)
            except Exception as e:  # network / schema problems must not kill the turn
                out = f"TOOL ERROR: {e}"
            harvest(out, cache, name)
            out = with_hints(name, args, out)
            # Harness-side shortcut: L2 tends to browse section trees (no cite_uid)
            # after a node search. Read the top hit's pages for it so citable
            # content is in hand after one model step.
            if name == "index_get_relevant_nodes":
                auto = auto_read_top(mcp, args, out)
                if auto:
                    harvest(auto, cache, "index_get_page_content")
                    out += "\n\nAUTO-FETCHED page content for the top result (citable):\n" + auto[:TOOL_RESULT_CHARS]
                    trace.append({"tool": "auto:index_get_page_content", "args": {}, "chars": len(auto), "sec": 0})
            trace.append({"tool": name, "args": args, "chars": len(out), "sec": round(time.time() - t, 1)})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out[:TOOL_RESULT_CHARS]})
        if final is not None:
            break

    if final is None:
        final = {"status": "partial" if cache else "no_evidence", "items": [], "note": "retrieval ended without finalize"}

    # resolve cite_uids -> cached items, best relevance first; if the model named
    # nothing but found things, keep the most recent few so generation has something
    chosen = sorted(final["items"], key=lambda x: -float(x.get("relevance_score", 0) or 0))
    items = [cache[c["cite_uid"]] for c in chosen if isinstance(c, dict) and c.get("cite_uid") in cache]
    selected_by = "finalize" if items else "none"
    if not items and cache:
        # L2 rarely calls finalize_retrieval in a long tool context. Let a strong
        # model pick from what was retrieved rather than guessing by recency.
        picked = select_with_model(selector, query, cache) if selector is not None else None
        if picked is not None and picked["items"]:
            items, selected_by = picked["items"], "selector"
            final["status"], final["note"] = picked["status"], picked["note"] or final.get("note", "")
        elif picked is not None and seeded:
            # Seeds are ranked searches (PubMed vector rank, HIRA relevance, MFDS exact
            # product): the first cached hits are the best ones. Better than nothing.
            items, selected_by = list(cache.values())[:3], "seed-rank"
            final["status"] = "partial"
        else:
            items, selected_by = list(cache.values())[-3:], "recency"
            final["status"] = "partial"
    return {
        "status": final["status"],
        "note": final.get("note", ""),
        "items": items[:8],
        "trace": trace,
        "tool_calls": calls,
        "cached_items": len(cache),
        "selected_by": selected_by,
        "seeded": seeded,
        "local_hits": len(local_hits),
        "elapsed": round(time.time() - t0, 1),
    }
