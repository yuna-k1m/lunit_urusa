import unittest
from pipeline.profiles import parse_profiles


def profile(identifier, focus):
    return {"id": identifier, "label": identifier, "perspective": f"Perspective {identifier}", "modified_query": f"Query {identifier}", "preserved_facts": ["fact"], "focus_areas": [focus]}


class ProfileValidationTest(unittest.TestCase):
    def test_parses_distinguishable_profiles(self):
        result = parse_profiles({"profiles": [profile("common", "likely causes"), profile("risk", "red flags")]}, 2)
        self.assertEqual([item.id for item in result], ["common", "risk"])

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_profiles({"profiles": [profile("same", "a"), profile("same", "b")]}, 2)

    def test_rejects_indistinguishable_focus(self):
        with self.assertRaisesRegex(ValueError, "distinguishable"):
            parse_profiles({"profiles": [profile("a", "Safety"), profile("b", "safety")]}, 2)


if __name__ == "__main__": unittest.main()
