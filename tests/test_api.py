import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app import DEFAULT_MODEL, app


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_models(self) -> None:
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], DEFAULT_MODEL)

    def test_chat_requires_messages(self) -> None:
        response = self.client.post("/v1/chat/completions", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "messages")

    def test_chat_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        self.assertEqual(response.status_code, 503)

    def test_chat_forwards_full_conversation(self) -> None:
        upstream_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "네."}}],
        }
        upstream_response = httpx.Response(
            200,
            json=upstream_body,
            request=httpx.Request("POST", "https://model.example/v1/chat/completions"),
        )
        messages = [
            {"role": "user", "content": "두통이 있어요."},
            {"role": "assistant", "content": "언제 시작했나요?"},
            {"role": "user", "content": "오늘 아침이요."},
        ]
        with (
            patch.dict(
                os.environ,
                {"LUNIT_FM_API_KEY": "lunit_test", "LUNIT_FM_API_URL": "https://model.example"},
                clear=True,
            ),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=upstream_response)) as post,
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "ignored", "messages": messages, "temperature": 0.2},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), upstream_body)
        forwarded = post.await_args.kwargs["json"]
        self.assertEqual(forwarded["messages"], messages)
        self.assertEqual(forwarded["model"], DEFAULT_MODEL)
        self.assertEqual(forwarded["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
