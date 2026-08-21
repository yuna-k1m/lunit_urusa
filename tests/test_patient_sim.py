import json
import unittest

from chat_models.base import ChatRequest
from chat_models.patient_sim import PatientSimStrategy


class FakeSimulator:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def generate_question(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("simulator failed")
        return "열도 있나요?"


class FakeSol:
    def __init__(self):
        self.calls = []

    async def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return "의사용 증상 정리" if len(self.calls) == 1 else "최종 환자 답변"


class FakeLunit:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, max_tokens=None):
        self.calls.append((messages, max_tokens))
        return "L2 감별진단"


class PatientSimTest(unittest.IsolatedAsyncioTestCase):
    async def test_runs_simulator_prep_diagnosis_and_finalizer(self):
        simulator, sol, lunit = FakeSimulator(), FakeSol(), FakeLunit()
        strategy = PatientSimStrategy(
            patient_simulator=simulator, lunit=lunit, sol=sol, pipeline_timeout=2
        )
        messages = [
            {"role": "user", "content": "머리가 아파요"},
            {"role": "assistant", "content": "언제부터인가요?"},
            {"role": "user", "content": "이틀 전부터요"},
        ]
        result = await strategy.complete(
            ChatRequest(messages=messages, original_payload={"max_tokens": 700})
        )
        self.assertEqual(result.content, "최종 환자 답변")
        self.assertEqual(simulator.calls, [messages])
        self.assertEqual(len(sol.calls), 2)
        self.assertIn("열도 있나요?", sol.calls[0]["input_text"])
        self.assertIn("L2 감별진단", sol.calls[1]["input_text"])
        self.assertEqual(lunit.calls[0][1], 700)
        self.assertEqual(result.metadata["finalizer"], "sol")

    async def test_failure_falls_back_to_original_history(self):
        lunit = FakeLunit()
        strategy = PatientSimStrategy(
            patient_simulator=FakeSimulator(fail=True), lunit=lunit,
            sol=FakeSol(), pipeline_timeout=2,
        )
        messages = [{"role": "user", "content": "Question"}]
        result = await strategy.complete(ChatRequest(messages=messages))
        self.assertEqual(result.metadata["fallback"], "direct_l2")
        self.assertEqual(lunit.calls[0][0], messages)


if __name__ == "__main__":
    unittest.main()
