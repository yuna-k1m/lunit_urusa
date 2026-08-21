import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from clients.health_db import HealthDBTool, TOOL_NAME
from lunit_health_db import HealthDB


SCHEMA = """
CREATE TABLE knowledge (
  id INTEGER PRIMARY KEY, source TEXT NOT NULL, source_key TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '', content TEXT NOT NULL, url TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}', fetched_at TEXT NOT NULL,
  UNIQUE(source, source_key)
);
CREATE VIRTUAL TABLE search USING fts5(
  kind UNINDEXED, ref_id UNINDEXED, title, content,
  tokenize='unicode61 remove_diacritics 2'
);
"""


class HealthDBToolTest(unittest.TestCase):
    def make_database(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO knowledge VALUES (1, 'guideline', 'g1', 'Hypertension guideline', "
            "'pregnancy blood pressure catalog', 'https://example.test', '{}', '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO search(kind,ref_id,title,content) VALUES ('knowledge','1',?,?)",
            ("Hypertension guideline", "pregnancy blood pressure catalog"),
        )
        # This row represents prohibited benchmark material. The tool must not return it.
        connection.execute(
            "INSERT INTO search(kind,ref_id,title,content) VALUES ('rubric','99',?,?)",
            ("Secret rubric", "pregnancy answer key"),
        )
        connection.commit()
        connection.close()

    def test_tool_exposes_only_knowledge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.db"
            self.make_database(path)
            result = json.loads(HealthDBTool(path).call({"query": "pregnancy", "limit": 10}))
            self.assertEqual([row["id"] for row in result["results"]], [1])
            self.assertEqual(result["results"][0]["cite_uid"], "health-db:1")
            self.assertNotIn("answer key", json.dumps(result))

    def test_health_db_handles_fts_punctuation_as_literal_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.db"
            self.make_database(path)
            with HealthDB(path) as database:
                self.assertTrue(database.search('pregnancy: "blood', kind="knowledge"))

    def test_tool_contract_and_missing_database(self):
        self.assertEqual(TOOL_NAME, "health_db_search")
        result = json.loads(HealthDBTool("/definitely/missing.db").call({"query": "pain"}))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
