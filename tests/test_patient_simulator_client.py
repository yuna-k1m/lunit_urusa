import unittest
from unittest.mock import patch

import httpx

from clients.patient_simulator import PatientSimulatorClient


class FakeAsyncClient:
    responses = []
    payloads = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, *, headers, json):
        self.payloads.append(json)
        status, body = self.responses.pop(0)
        request = httpx.Request("POST", url)
        return httpx.Response(status, json=body, request=request)


def completion(text):
    return {"choices": [{"message": {"content": text}}]}


class PatientSimulatorClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncClient.responses = []
        FakeAsyncClient.payloads = []
        self.client = PatientSimulatorClient(
            base_url="https://patient.example", api_key="lunit_test",
            model="patient-simulator-ko", timeout=10,
        )

    @patch("clients.patient_simulator.asyncio.sleep", return_value=None)
    @patch("clients.patient_simulator.httpx.AsyncClient", FakeAsyncClient)
    async def test_retries_502_without_changing_history(self, _sleep):
        FakeAsyncClient.responses = [(502, {}), (200, completion("다음 질문"))]
        history = [{"role": "user", "content": "원본"}]
        result = await self.client.generate_question(history)
        self.assertEqual(result, "다음 질문")
        self.assertEqual([call["messages"] for call in FakeAsyncClient.payloads], [history, history])

    @patch("clients.patient_simulator.asyncio.sleep", return_value=None)
    @patch("clients.patient_simulator.httpx.AsyncClient", FakeAsyncClient)
    async def test_404_restarts_with_empty_history(self, _sleep):
        FakeAsyncClient.responses = [(404, {}), (200, completion("새 질문"))]
        history = [{"role": "user", "content": "expired conversation"}]
        result = await self.client.generate_question(history)
        self.assertEqual(result, "새 질문")
        self.assertEqual(FakeAsyncClient.payloads[0]["messages"], history)
        self.assertEqual(FakeAsyncClient.payloads[1]["messages"], [])


if __name__ == "__main__":
    unittest.main()
