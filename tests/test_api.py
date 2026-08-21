"""API contract tests for the conversation driver (no network).

    python -m unittest discover -s tests -v
"""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import engine, main


def fake_answer(client, messages, *, temperature=0.3, max_tokens=2048, planner=None):
    return {
        "answer": f"echo:{messages[-1]['content']} (n={len(messages)})",
        "plan": dict(engine.PLAN_DEFAULTS),
        "notes": {},
        "timings": {},
    }


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def test_models(self) -> None:
        r = self.client.get("/v1/models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["object"], "list")
        self.assertTrue(r.json()["data"][0]["id"])

    def test_chat_shape_and_full_history(self) -> None:
        messages = [
            {"role": "user", "content": "첫 질문"},
            {"role": "assistant", "content": "첫 답변"},
            {"role": "user", "content": "후속 질문"},
        ]
        with patch.object(main.engine, "answer", fake_answer):
            r = self.client.post(
                "/v1/chat/completions",
                json={"model": "evaluator-alias", "messages": messages, "temperature": 0},
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(body["choices"][0]["message"]["content"], "echo:후속 질문 (n=3)")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_content_parts_are_flattened(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        with patch.object(main.engine, "answer", fake_answer):
            r = self.client.post("/v1/chat/completions", json={"messages": messages})
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "echo:hi (n=1)")


class KeyResolutionTest(unittest.TestCase):
    def test_env_wins_over_file(self) -> None:
        with patch.dict(os.environ, {"LUNIT_FM_API_KEY": "lunit_env"}, clear=False):
            self.assertEqual(engine.resolve_key("LUNIT_FM_API_KEY"), "lunit_env")

    def test_bundled_files_present(self) -> None:
        # The submission contract: both key files exist in the tree and are non-empty.
        for name, path in engine.KEY_FILES.items():
            self.assertTrue(path.exists(), f"{path.name} missing ({name})")
            self.assertTrue(path.read_text(encoding="utf-8").strip(), f"{path.name} empty")


class AssembleTest(unittest.TestCase):
    def test_questions_appended_when_missing(self) -> None:
        plan = dict(engine.PLAN_DEFAULTS)
        plan.update(context_sufficient=False, questions=["How old is the child?"], questions_intro="Tell me:")
        text, notes = engine.assemble("Some answer.", plan)
        self.assertTrue(notes["appended_questions"])
        self.assertTrue(text.endswith("- How old is the child?"))

    def test_directive_prepended_when_missing(self) -> None:
        plan = dict(engine.PLAN_DEFAULTS)
        plan.update(urgency="emergent", emergency_directive="Call 119 now.")
        text, notes = engine.assemble("Keep him calm.", plan)
        self.assertTrue(notes["prepended_directive"])
        self.assertTrue(text.startswith("**Call 119 now.**"))


if __name__ == "__main__":
    unittest.main()
