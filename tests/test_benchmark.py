import tempfile
import unittest
from pathlib import Path

from benchmark.run_healthbench import calculate_score, fixed_sample, load_grader_template


class BenchmarkTest(unittest.TestCase):
    def test_fixed_sample_is_independent_of_row_order(self):
        rows = [{"prompt_id": value} for value in ("d", "a", "c", "b")]
        self.assertEqual([row["prompt_id"] for row in fixed_sample(rows, 3)], ["a", "b", "c"])
        self.assertEqual(
            [row["prompt_id"] for row in fixed_sample(list(reversed(rows)), 3)],
            ["a", "b", "c"],
        )

    def test_score_includes_triggered_negative_points(self):
        rubrics = [{"points": 10}, {"points": 5}, {"points": -5}]
        grades = [{"criteria_met": True}, {"criteria_met": False}, {"criteria_met": True}]
        self.assertEqual(calculate_score(rubrics, grades), 5 / 15)

    def test_loads_stripped_template_without_importing_module(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grader.py"
            path.write_text('GRADER_TEMPLATE = "  template  ".strip()\n', encoding="utf-8")
            self.assertEqual(load_grader_template(path), "template")


if __name__ == "__main__":
    unittest.main()
