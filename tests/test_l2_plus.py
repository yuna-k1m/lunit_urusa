import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.retrieval import run_retrieval
from chat_models.factory import build_registry
from clients.l2_plus import L2PlusClient
from config import Settings


class FakeHealthDB:
    available = True

    def __init__(self, events):
        self.events = events

    def search(self, query, limit=5):
        self.events.append(("local", query, limit))
        return [{
            "cite_uid": "health-db:1",
            "title": "Hypertension guideline",
            "source_type": "local_health_catalog",
            "source": "lunit_mcp:index_list_documents",
            "url": "",
            "content": json.dumps({"node_id": "guideline-node-1"}),
        }]


class FakeMCP:
    def __init__(self):
        self.events = []
        self.health_db = FakeHealthDB(self.events)

    def tools(self):
        self.events.append(("remote", "tools/list"))
        return []

    def call(self, name, arguments):
        self.events.append(("remote", name, arguments))
        if name == "index_get_relevant_nodes":
            return json.dumps([{"doc_id": "guideline-node-1", "range": [3, 5]}])
        if name == "index_get_page_content":
            return json.dumps({
                "cite_uid": "mcp-page-3",
                "title": "Hypertension guideline",
                "content": "Citable guideline page content",
            })
        raise AssertionError(f"unexpected MCP call: {name}")


class NeverCalledL2:
    def chat_message(self, *args, **kwargs):
        raise AssertionError("deterministic local routing should avoid the L2 tool loop")


class L2PlusTest(unittest.TestCase):
    def test_local_database_runs_before_targeted_mcp_and_skips_broad_loop(self):
        mcp = FakeMCP()
        result = run_retrieval(
            NeverCalledL2(), "pregnancy hypertension", mcp=mcp, local_first=True
        )
        self.assertEqual(mcp.events[0][0], "local")
        self.assertEqual(mcp.events[1][1], "index_get_relevant_nodes")
        self.assertEqual(mcp.events[2][1], "index_get_page_content")
        self.assertNotIn(("remote", "tools/list"), mcp.events)
        self.assertEqual(result["local_hits"], 1)
        self.assertTrue(result["seeded"])
        self.assertEqual(result["items"][0]["cite_uid"], "mcp-page-3")

    def test_client_adapts_engine_answer_to_existing_async_contract(self):
        internal = object()
        client = L2PlusClient(
            base_url="https://example.test", api_key="key", model="L2", timeout=10,
            client=internal, planner=None,
        )
        messages = [{"role": "user", "content": "question"}]
        with patch("clients.l2_plus.engine.answer", return_value={"answer": "grounded answer"}) as answer:
            result = asyncio.run(client.complete(messages, max_tokens=123))
        self.assertEqual(result, "grounded answer")
        answer.assert_called_once_with(
            internal, messages, max_tokens=123, planner=None, local_first=True
        )

    def test_factory_selects_l2_plus_without_changing_baseline(self):
        settings = replace(
            Settings.from_env(), l2_backend="l2_plus",
            lunit_api_key="test", openai_api_key="test",
        )
        registry = build_registry(settings)
        with patch("clients.l2_plus.engine.planner_from_env", return_value=None):
            selected = registry.create("direct_l2")
        baseline = registry.create("baseline_l2")
        self.assertIsInstance(selected.lunit, L2PlusClient)
        self.assertNotIsInstance(baseline.lunit, L2PlusClient)
        self.assertTrue(registry.create("siusiubeom_h4").local_first)


if __name__ == "__main__":
    unittest.main()
