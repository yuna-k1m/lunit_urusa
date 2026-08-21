"""Model-safe local tool for the vendored Lunit health database.

Only the ``knowledge`` table is reachable through this adapter. HealthBench
examples, rubrics, ideal answers, and model outputs are deliberately excluded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lunit_health_db import HealthDB


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "resources" / "lunit_health.db"
TOOL_NAME = "health_db_search"

TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Search the local Lunit health knowledge catalog for relevant medical guidelines and data "
        "sources. Use this to discover relevant sources. It never returns benchmark examples, "
        "rubrics, ideal answers, or model outputs."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A short, self-contained medical search query.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


class HealthDBTool:
    def __init__(self, path: str | Path | None = None) -> None:
        configured = os.getenv("LUNIT_HEALTH_DB_PATH")
        self.path = Path(path or configured or DEFAULT_DB_PATH)

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return knowledge-only records for driver-side local-first routing."""
        if not isinstance(query, str) or not query.strip() or not self.available:
            return []
        limit = max(1, min(int(limit), 10))
        with HealthDB(self.path) as database:
            hits = database.search(query, limit=limit, kind="knowledge")
            results = []
            for hit in hits:
                record = database.knowledge(hit["ref_id"])
                if record is None:
                    continue
                results.append(
                    {
                        "cite_uid": f"health-db:{record['id']}",
                        "id": record["id"],
                        "title": record["title"],
                        "source_type": "local_health_catalog",
                        "source": record["source"],
                        "url": record["url"],
                        "content": record["content"],
                        "score": hit["score"],
                    }
                )
        return results

    def call(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "query must be a non-empty string"})
        try:
            limit = max(1, min(int(arguments.get("limit", 5)), 10))
        except (TypeError, ValueError):
            limit = 5
        if not self.available:
            return json.dumps({"error": f"health database not found: {self.path}"})
        results = self.search(query, limit)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
