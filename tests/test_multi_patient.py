import asyncio
import unittest
from chat_models.base import ChatRequest
from chat_models.multi_patient import MultiPatientStrategy


def profiles():
    return {"profiles": [
        {"id": "common", "label": "Common", "perspective": "Likely causes", "modified_query": "원래 사실을 유지한 일반 관점 질문", "preserved_facts": ["증상"], "focus_areas": ["common causes"]},
        {"id": "risk", "label": "Risk", "perspective": "Red flags", "modified_query": "원래 사실을 유지한 위험 관점 질문", "preserved_facts": ["증상"], "focus_areas": ["red flags"]},
        {"id": "context", "label": "Context", "perspective": "Missing context", "modified_query": "원래 사실을 유지한 맥락 관점 질문", "preserved_facts": ["증상"], "focus_areas": ["missing context"]},
    ]}


class FakeSol:
    def __init__(self, fail_profiles=False, fail_aggregation=False):
        self.fail_profiles = fail_profiles
        self.fail_aggregation = fail_aggregation
        self.aggregation_input = None
    async def generate_json(self, **kwargs):
        if self.fail_profiles: raise RuntimeError("profile failure")
        return profiles()
    async def generate_text(self, **kwargs):
        if self.fail_aggregation: raise RuntimeError("aggregation failure")
        self.aggregation_input = kwargs["input_text"]
        return "최종 통합 답변"


class FakeLunit:
    model = "Lunit/L2-preview"
    def __init__(self, fail_on=None):
        self.calls, self.active, self.max_active, self.fail_on = [], 0, 0, fail_on
    async def complete(self, messages):
        self.calls.append(messages)
        text = messages[-1]["content"]
        if self.fail_on and self.fail_on in text: raise RuntimeError("branch failure")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return f"L2 answer {len(self.calls)}"


def strategy(lunit, sol, *, minimum=2, max_parallel=2):
    return MultiPatientStrategy(lunit=lunit, sol=sol, profile_count=3, max_parallel=max_parallel, minimum_successes=minimum, pipeline_timeout=2)


class MultiPatientTest(unittest.IsolatedAsyncioTestCase):
    async def test_fans_out_with_concurrency_limit_and_aggregates(self):
        lunit, sol = FakeLunit(), FakeSol()
        result = await strategy(lunit, sol).complete(ChatRequest(messages=[{"role": "user", "content": "두통이 있어요"}]))
        self.assertEqual(result.content, "최종 통합 답변")
        self.assertEqual(len(lunit.calls), 3)
        self.assertLessEqual(lunit.max_active, 2)
        self.assertIn("두통이 있어요", sol.aggregation_input)
        self.assertEqual(result.metadata["profiles_succeeded"], 3)

    async def test_aggregates_after_one_branch_failure(self):
        result = await strategy(FakeLunit(fail_on="Red flags"), FakeSol()).complete(ChatRequest(messages=[{"role": "user", "content": "Question"}]))
        self.assertEqual(result.metadata["profiles_succeeded"], 2)

    async def test_profile_failure_falls_back_to_direct_l2(self):
        lunit, original = FakeLunit(), [{"role": "user", "content": "Original"}]
        result = await strategy(lunit, FakeSol(fail_profiles=True)).complete(ChatRequest(messages=original))
        self.assertEqual(result.metadata["fallback"], "direct_l2")
        self.assertEqual(lunit.calls, [original])

    async def test_sol_aggregation_failure_falls_back_to_direct_l2(self):
        lunit = FakeLunit()
        result = await strategy(lunit, FakeSol(fail_aggregation=True)).complete(ChatRequest(messages=[{"role": "user", "content": "Question"}]))
        self.assertEqual(len(lunit.calls), 4)
        self.assertEqual(result.metadata["fallback"], "direct_l2")


if __name__ == "__main__": unittest.main()
