import unittest

from tools.run_eval import calculate_score, pick_sample


class UnifiedEvalTest(unittest.TestCase):
    def test_fixed100_is_stable_and_ignores_n(self):
        rows = [{"prompt_id": f"{index:03d}", "prompt": []} for index in reversed(range(120))]
        selected = pick_sample(rows, "fixed100", n=3, seed=99, slice_="all")
        self.assertEqual(len(selected), 100)
        self.assertEqual(selected[0]["prompt_id"], "000")
        self.assertEqual(selected[-1]["prompt_id"], "099")

    def test_score_preserves_negative_penalties(self):
        rubrics = [{"points": 10}, {"points": -5}]
        grades = [{"criteria_met": True}, {"criteria_met": True}]
        self.assertEqual(calculate_score(rubrics, grades), 0.5)


if __name__ == "__main__":
    unittest.main()
