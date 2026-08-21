"""API contract tests for the conversation driver (no network).

    python -m unittest discover -s tests -v
"""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from chat_models.base import ChatResult

from app import engine, main


class FakeStrategy:
    name = "fake"

    def __init__(self) -> None:
        self.request = None

    async def complete(self, request):
        self.request = request
        return ChatResult(content="통합 답변", model=self.name)


def fake_answer(client, messages, *, temperature=0.3, max_tokens=2048, planner=None):
    return {
        "answer": f"echo:{messages[-1]['content']} (n={len(messages)})",
        "plan": dict(engine.PLAN_DEFAULTS),
        "notes": {},
        "timings": {},
    }


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        # Exercise the legacy engine branch in these two transport tests even
        # when the submitted image has a strategy selected via Docker ENV.
        self.environment = patch.dict(os.environ, {"MODEL_STRATEGY": ""}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
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

    def test_selected_strategy_preserves_full_history(self) -> None:
        strategy = FakeStrategy()
        messages = [
            {"role": "user", "content": "첫 질문"},
            {"role": "assistant", "content": "첫 답변"},
            {"role": "user", "content": "그 약은요?"},
        ]
        main.app.state.strategy_override = strategy
        try:
            response = self.client.post("/v1/chat/completions", json={"messages": messages})
        finally:
            del main.app.state.strategy_override
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "통합 답변")
        self.assertEqual(strategy.request.messages, messages)

    def test_chat_never_refuses_empty_messages_or_streaming(self) -> None:
        # The evaluator treats any non-2xx as a failed trial: empty history gets a
        # greeting, stream=true is served as SSE with the full answer.
        with patch.object(main.engine, "answer", fake_answer):
            self.assertEqual(self.client.post("/v1/chat/completions", json={}).status_code, 200)
            response = self.client.post(
                "/v1/chat/completions",
                json={"stream": True, "messages": [{"role": "user", "content": "Hello"}]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("echo:Hello", response.text)
        self.assertTrue(response.text.strip().endswith("data: [DONE]"))


class KeyResolutionTest(unittest.TestCase):
    def test_env_wins_over_file(self) -> None:
        with patch.dict(os.environ, {"LUNIT_FM_API_KEY": "lunit_env"}, clear=False):
            self.assertEqual(engine.resolve_key("LUNIT_FM_API_KEY"), "lunit_env")

    def test_sanitized_fork_has_no_bundled_keys(self) -> None:
        # The public fork receives credentials only through environment variables.
        for path in engine.KEY_FILES.values():
            self.assertFalse(path.exists(), f"{path.name} must not be committed")


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


class SubmissionDefaultsTest(unittest.TestCase):
    def test_dockerfile_selects_direct_l2_with_an_l2_final_strategy(self) -> None:
        # Submission rule: final text must come from L2. siusiubeom_h4 satisfies
        # that rule while its internal backend performs local-first retrieval.
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.joinpath("Dockerfile")
        if not path.exists():
            self.skipTest("Dockerfile is validated before the tests are mounted into the image")
        text = path.read_text(encoding="utf-8")
        self.assertIn("MODEL_STRATEGY=siusiubeom_h4", text)
        self.assertIn("L2_BACKEND=direct_l2", text)
        self.assertNotIn("MODEL_STRATEGY=multi_patient", text)
        self.assertNotIn("DRIVER_ENGINE=probe", text)
