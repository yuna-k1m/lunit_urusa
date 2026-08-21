import asyncio
import json
import unittest

from chat_models.base import ChatRequest
from chat_models.demographics import DemographicsStrategy
from pipeline.demographics import predefined_demographics


class FakeSol:
    def __init__(self, fail_aggregation=False):
        self.fail_aggregation = fail_aggregation
        self.aggregation_input = None
        self.max_output_tokens = None

    async def generate_json(self, **kwargs):
        raise AssertionError("predefined profiles must not call Sol")

    async def generate_text(self, **kwargs):
        if self.fail_aggregation:
            raise RuntimeError("aggregation failure")
        self.aggregation_input = kwargs["input_text"]
        self.max_output_tokens = kwargs.get("max_output_tokens")
        return "Final demographic-safe answer"


class FakeLunit:
    model = "Lunit/L2-preview"

    def __init__(self, fail_on=None):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.fail_on = fail_on
        self.max_tokens = []

    async def complete(self, messages, *, max_tokens=None):
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        text = messages[-1]["content"]
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("branch failure")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return f"L2 answer {len(self.calls)}"


def strategy(lunit, sol, *, minimum=2, max_parallel=2):
    return DemographicsStrategy(
        lunit=lunit, sol=sol, profile_count=3, max_parallel=max_parallel,
        minimum_successes=minimum, pipeline_timeout=2,
    )


class DemographicsTest(unittest.IsolatedAsyncioTestCase):
    async def test_fans_out_and_aggregates_hypothetical_profiles(self):
        lunit, sol = FakeLunit(), FakeSol()
        original = [{"role": "user", "content": "I have a headache"}]
        result = await strategy(lunit, sol).complete(ChatRequest(
            messages=original, original_payload={"max_tokens": 4096}
        ))
        self.assertEqual(result.content, "Final demographic-safe answer")
        self.assertEqual(result.metadata["profile_type"], "demographic_sensitivity")
        self.assertEqual(len(lunit.calls), 3)
        self.assertLessEqual(lunit.max_active, 2)
        self.assertEqual(lunit.max_tokens, [4096, 4096, 4096])
        self.assertEqual(sol.max_output_tokens, 4096)
        self.assertIn("hypothetical", lunit.calls[0][0]["content"].lower())
        aggregation = json.loads(sol.aggregation_input)
        self.assertEqual(aggregation["raw_input"], original)
        self.assertEqual(len(aggregation["l2_answers"]), 3)

    async def test_one_failed_branch_is_tolerated(self):
        result = await strategy(
            FakeLunit(fail_on="pregnancy and reproductive"), FakeSol()
        ).complete(ChatRequest(messages=[{"role": "user", "content": "Question"}]))
        self.assertEqual(result.metadata["profiles_succeeded"], 2)

    async def test_aggregation_failure_falls_back_to_direct_l2(self):
        lunit = FakeLunit()
        result = await strategy(lunit, FakeSol(fail_aggregation=True)).complete(
            ChatRequest(messages=[{"role": "user", "content": "Question"}])
        )
        self.assertEqual(len(lunit.calls), 4)
        self.assertEqual(result.metadata["fallback"], "direct_l2")


class PredefinedDemographicsTest(unittest.TestCase):
    def test_default_three_span_child_adult_and_older_adult(self):
        profiles = predefined_demographics("raw conversation", 3)
        self.assertEqual([p.id for p in profiles], ["child_boy", "young_woman", "older_woman"])
        self.assertTrue(all(p.modified_query == "raw conversation" for p in profiles))

    def test_sets_are_defined_for_one_through_five(self):
        for count in range(1, 6):
            profiles = predefined_demographics("query", count)
            self.assertEqual(len(profiles), count)
            self.assertEqual(len({p.id for p in profiles}), count)

    def test_rejects_unsupported_profile_count(self):
        with self.assertRaisesRegex(ValueError, "1 through 5"):
            predefined_demographics("query", 6)


if __name__ == "__main__":
    unittest.main()
