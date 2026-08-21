import unittest

from fastapi.testclient import TestClient

from app import MODEL_ID, app


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_models(self) -> None:
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(response.json()["data"][0]["id"], MODEL_ID)

    def test_chat_requires_messages(self) -> None:
        response = self.client.post("/v1/chat/completions", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_chat_returns_static_openai_response(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], MODEL_ID)
        self.assertEqual(
            body["choices"][0]["message"]["content"],
            "Submission container is running.",
        )


if __name__ == "__main__":
    unittest.main()
