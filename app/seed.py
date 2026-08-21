"""Deterministic retrieval seeds driven by the planner's structured hints.

L2 navigates the MCP tools poorly: it cannot reliably find the right guideline
section by semantic search within budget, and it rarely completes multi-call
chains (law search -> list articles -> get article). The planner (a strong model)
knows *what* to look up; the harness can perform the obvious first calls itself:

    guideline_title -> resolve doc_id -> node search scoped to that doc -> read pages
    drug_inn        -> adr_retrieve_drug_info
    pubmed_query    -> rag_vector_query(pubmed_abstracts)
    hira_query      -> hira_updates_search
    kcd_name        -> kcd_search_codes
    statutes        -> openapi_law_search -> list_articles(contains) -> get_article

Every result is harvested into the cite_uid cache. If the seeds produce citable
items, the L2 retrieval loop is skipped entirely (faster, and more reliable);
otherwise it runs with what remains of the budget.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from app.retrieval import MCPClient, harvest

HINT_KEYS = ("guideline_title", "guideline_item", "drug_inn", "pubmed_query", "hira_query",
             "kcd_name", "statutes", "law_keyword", "mfds_drug")


def normalize_hints(raw: Any) -> dict:
    h = {k: "" for k in HINT_KEYS}
    h["statutes"] = []
    if not isinstance(raw, dict):
        return h
    for k in HINT_KEYS:
        v = raw.get(k)
        if k == "statutes":
            if isinstance(v, str):
                v = [v]
            h[k] = [str(x).strip() for x in (v or []) if str(x).strip()][:2]
        elif v:
            h[k] = str(v).strip()
    return h


def has_any(h: dict) -> bool:
    return any(h[k] for k in HINT_KEYS)


# ----------------------------------------------------------------- seed steps

def _call(mcp: MCPClient, name: str, args: dict, cache: dict, trace: list) -> str:
    try:
        out = mcp.call(name, args)
    except Exception as e:  # seeds are best effort
        out = f"TOOL ERROR: {e}"
    harvest(out, cache, name)
    trace.append({"tool": f"seed:{name}", "args": args, "chars": len(out), "sec": 0})
    return out


def seed_guideline(mcp: MCPClient, title: str, item: str, cache: dict, trace: list,
                   known_titles: list[str]) -> None:
    # resolve the title against the corpus; the planner copies from our list, but be tolerant
    best = difflib.get_close_matches(title, known_titles, n=1, cutoff=0.5)
    want = best[0] if best else title
    out = _call(mcp, "index_list_documents", {"corpus_tag": "guideline", "query": want, "limit": 5}, cache, trace)
    try:
        docs = json.loads(out).get("results", [])
    except (json.JSONDecodeError, ValueError, AttributeError):
        return
    doc = None
    for d in docs:
        if difflib.SequenceMatcher(None, d.get("title", ""), want).ratio() > 0.8:
            doc = d
            break
    if doc is None and docs:
        doc = docs[0]
    if doc is None:
        return
    seed_guideline_node(mcp, str(doc["node_id"]), item or title, cache, trace)


def seed_guideline_node(mcp: MCPClient, node_id: str, query: str,
                        cache: dict, trace: list) -> None:
    """Read citable pages using a document id discovered in the local DB."""
    out = _call(mcp, "index_get_relevant_nodes",
                {"corpus_tag": "guideline", "node_id": node_id, "query": query, "k": 4}, cache, trace)
    try:
        nodes = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        return
    seen: set[tuple] = set()
    for n in (nodes if isinstance(nodes, list) else [])[:2]:
        rng = n.get("range")
        if not (isinstance(rng, list) and len(rng) == 2):
            continue
        a, b = int(rng[0]), min(int(rng[1]), int(rng[0]) + 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        _call(mcp, "index_get_page_content",
              {"corpus_tag": "guideline", "doc_id": n.get("doc_id", node_id), "start_page": a, "end_page": b},
              cache, trace)


def seed_law(mcp: MCPClient, statutes: list[str], keyword: str, cache: dict, trace: list) -> None:
    for name in statutes[:2]:
        out = _call(mcp, "openapi_law_search", {"query": name, "kind": "law"}, cache, trace)
        msts = re.findall(r"mst=(\d+)\s*·\s*([^·\n]+)", out)
        # prefer the statute itself over its 시행령/시행규칙
        msts.sort(key=lambda m: ("시행" in m[1], len(m[1])))
        for mst, _ in msts[:1]:
            la = _call(mcp, "openapi_law_list_articles", {"mst": mst, "contains": keyword or None}, cache, trace)
            if la.startswith("TOOL ERROR"):
                continue
            keys = re.findall(r'"(?:article_key|key)"\s*:\s*"([^"]+)"', la)[:3]
            if keys:
                _call(mcp, "openapi_law_get_article", {"mst": mst, "article_keys": keys}, cache, trace)


def run_seeds(mcp: MCPClient, hints: dict, cache: dict, trace: list, known_titles: list[str]) -> None:
    if hints["guideline_title"]:
        seed_guideline(mcp, hints["guideline_title"], hints["guideline_item"], cache, trace, known_titles)
    if hints["drug_inn"]:
        _call(mcp, "adr_retrieve_drug_info", {"drug_name": hints["drug_inn"]}, cache, trace)
    if hints["mfds_drug"]:
        _call(mcp, "openapi_mfds_get_drug_indication", {"drug_name": hints["mfds_drug"], "num_rows": 3}, cache, trace)
        _call(mcp, "openapi_mfds_check_drug_permission", {"drug_name": hints["mfds_drug"], "num_rows": 3}, cache, trace)
    if hints["pubmed_query"]:
        _call(mcp, "rag_vector_query",
              {"collection_name": "pubmed_abstracts", "query": hints["pubmed_query"], "top_k": 5}, cache, trace)
    if hints["hira_query"]:
        _call(mcp, "hira_updates_search",
              {"query": hints["hira_query"], "current_only": True, "limit": 5, "search_mode": "both"}, cache, trace)
        # HIRA notices name drugs by English INN ("Empagliflozin 경구제(품명: 자디앙정)"):
        # a Korean transliteration alone misses them.
        if hints["drug_inn"]:
            _call(mcp, "hira_updates_search",
                  {"query": f"{hints['drug_inn']} 급여", "current_only": True, "limit": 5, "search_mode": "both"},
                  cache, trace)
    if hints["kcd_name"]:
        _call(mcp, "kcd_search_codes", {"name": hints["kcd_name"], "lang": "kor", "top_k": 5}, cache, trace)
    if hints["statutes"]:
        seed_law(mcp, hints["statutes"], hints["law_keyword"], cache, trace)


# --------------------------------------------------- unverifiable specifics

UNVERIFIED = re.compile(
    r"제\s?\d+\s?조|시행령\s*제?\s*\d+|시행규칙\s*제?\s*\d+|고시\s*제\s?\d{4}-\d+호|"
    r"\bClass\s*(?:1|2a|2b|3|I{1,3}|IIa|IIb)\b|권고\s*등급\s*[:：]?\s*(?:1|2a|2b|3|I{1,3})|"
    r"근거\s*수준\s*[:：]?\s*[A-C](?:-(?:R|NR|LD|EO))?|\bLevel\s*[A-C]\b|\bCategory\s*(?:1|2A|2B|3)\b|"
    r"PMID[:\s]*\d+|\d+\s*일\s*이내",
)


def unverified_specifics(text: str) -> list[str]:
    """Specific, checkable claims that should not appear without a source."""
    return sorted(set(m.group(0) for m in UNVERIFIED.finditer(text)))
