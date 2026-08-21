import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app import DEFAULT_MODEL, app


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_models(self) -> None:
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(response.json()["data"][0]["id"], DEFAULT_MODEL)

    def test_chat_requires_messages(self) -> None:
        response = self.client.post("/v1/chat/completions", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_chat_requires_server_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["type"], "server_error")

    def test_chat_forwards_full_history_and_pins_model(self) -> None:
        messages = [
            {"role": "user", "content": "첫 질문"},
            {"role": "assistant", "content": "첫 답변"},
            {"role": "user", "content": "후속 질문"},
        ]
        result = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "답변"}}
            ],
        }
        upstream = httpx.Response(
            200,
            content=json.dumps(result).encode(),
            headers={"content-type": "application/json"},
        )

        with patch.dict(os.environ, {"LUNIT_FM_API_KEY": "lunit_test"}, clear=True):
            with patch("app.forward_chat", AsyncMock(return_value=upstream)) as forward:
                response = self.client.post(
                    "/v1/chat/completions",
                    json={"model": "evaluator-alias", "messages": messages},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)
        sent_payload, sent_key = forward.await_args.args
        self.assertEqual(sent_key, "lunit_test")
        self.assertEqual(sent_payload["model"], DEFAULT_MODEL)
        self.assertEqual(sent_payload["messages"], messages)


if __name__ == "__main__":
    unittest.main()
