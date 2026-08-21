from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class HealthDB:
    """Small read-only query interface for the Lunit health SQLite database."""

    def __init__(self, path: str | Path):
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "HealthDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(self, query: str, limit: int = 8, kind: str | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        where = "search MATCH ?"
        args: list[Any] = [query]
        if kind:
            where += " AND kind = ?"
            args.append(kind)
        args.append(max(1, min(limit, 100)))
        sql = (
            "SELECT kind, ref_id, title, snippet(search, 3, '[', ']', ' … ', 24) AS snippet, "
            f"bm25(search) AS score FROM search WHERE {where} ORDER BY score LIMIT ?"
        )
        try:
            rows = self.connection.execute(sql, args)
        except sqlite3.OperationalError:
            # Treat punctuation-heavy model input as literal words rather than FTS syntax.
            words = [word.replace('"', '""') for word in query.split() if word]
            if not words:
                return []
            args[0] = " OR ".join(f'"{word}"' for word in words)
            rows = self.connection.execute(sql, args)
        return [dict(row) for row in rows]

    def knowledge(self, ref_id: str | int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT id, source, source_key, title, content, url, metadata_json, fetched_at "
            "FROM knowledge WHERE id = ?",
            (ref_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def example(self, prompt_id: str) -> dict[str, Any] | None:
        """Offline analysis API. Never expose this method as a model tool."""
        row = self.connection.execute(
            "SELECT * FROM examples WHERE prompt_id = ?", (prompt_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["prompt"] = json.loads(result.pop("conversation_json"))
        result["example_tags"] = json.loads(result.pop("tags_json"))
        result["rubrics"] = [
            {
                "criterion": rubric["criterion"],
                "points": rubric["points"],
                "tags": json.loads(rubric["tags_json"]),
            }
            for rubric in self.connection.execute(
                "SELECT criterion, points, tags_json FROM rubrics "
                "WHERE prompt_id = ? ORDER BY id",
                (prompt_id,),
            )
        ]
        return result

    def knowledge_for(self, prompt_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT k.*, ek.relevance FROM knowledge k "
            "JOIN example_knowledge ek ON k.id=ek.knowledge_id "
            "WHERE ek.prompt_id=? ORDER BY ek.relevance DESC",
            (prompt_id,),
        )
        return [dict(row) for row in rows]
