import unittest
from dataclasses import replace

from chat_models.factory import build_registry
from chat_models.registry import StrategyRegistry
from config import Settings


class Item: name = "item"


class RegistryTest(unittest.TestCase):
    def test_register_create_and_list(self):
        registry = StrategyRegistry()
        registry.register("item", Item)
        self.assertIsInstance(registry.create("item"), Item)
        self.assertEqual(registry.names(), ["item"])

    def test_unknown_strategy_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "unknown MODEL_STRATEGY"):
            StrategyRegistry().create("missing")

    def test_application_exposes_baseline_and_multi_patient_models(self):
        settings = Settings.from_env()
        registry = build_registry(replace(settings, lunit_api_key="test", openai_api_key="test"))
        self.assertIn("baseline_l2", registry.names())
        self.assertIn("demographics", registry.names())
        self.assertIn("demographics_sol", registry.names())
        self.assertIn("multi_patient_sol", registry.names())
        self.assertIn("patient-sim", registry.names())
        self.assertIn("siusiubeom_h4", registry.names())

    def test_unknown_l2_backend_is_actionable_when_strategy_is_created(self):
        settings = replace(Settings.from_env(), l2_backend="missing")
        registry = build_registry(settings)
        with self.assertRaisesRegex(ValueError, "unknown L2_BACKEND"):
            registry.create("direct_l2")


if __name__ == "__main__": unittest.main()
