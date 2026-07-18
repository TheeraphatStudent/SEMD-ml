"""T08 (docs/refactoring-plan.md): direct unit coverage for tracking/registry.py's
Registry and tracking/promotion.py's Promotion, independent of the
ModelRegistryManager facade that tests/unit/test_model_registry.py already
exercises thoroughly end-to-end with a FakeMlflowClient. This locks the
decomposition boundary itself: Registry has no promotion-gate knowledge, and
Promotion depends on a Registry instance rather than reimplementing CRUD.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from tracking.promotion import GateResult, ModelValidationError, Promotion
from tracking.registry import ModelReference, ModelRegistryError, Registry


class RegistryUnavailableTests(unittest.TestCase):
    def test_require_available_raises_when_no_client(self):
        registry = Registry(client=None)
        registry.client = None  # force unavailable regardless of env MLflow config
        with self.assertRaises(ModelRegistryError):
            registry.require_available()

    def test_available_reflects_client_presence(self):
        registry = Registry(client=None)
        registry.client = None
        self.assertFalse(registry.available)
        registry.client = MagicMock()
        self.assertTrue(registry.available)

    def test_get_reference_requires_alias_or_version(self):
        registry = Registry(client=MagicMock())
        with self.assertRaises(ModelRegistryError):
            registry.get_reference()


class RegistryHasNoPromotionKnowledgeTests(unittest.TestCase):
    def test_registry_has_no_gate_or_promotion_methods(self):
        # T08 acceptance criterion: tracking run logging/registry CRUD works
        # independently of model promotion. Enforce it structurally: Registry
        # carries no gate-evaluation or promote/validate methods -- those live
        # on Promotion, which composes a Registry (the dependency direction is
        # one-way: Promotion -> Registry, never the reverse).
        registry_methods = {name for name in dir(Registry) if not name.startswith("__")}
        self.assertTrue(registry_methods.isdisjoint({
            "promote_candidate", "validate_candidate", "_evaluate_gates", "_compare_to_champion",
        }))

    def test_registry_does_not_import_promotion_at_module_level(self):
        import ast

        import tracking.registry as registry_module

        with open(registry_module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in top_level_imports:
            module = getattr(node, "module", None) or ""
            self.assertNotIn("promotion", module)


class PromotionGateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.registry = Registry(client=self.client)
        self.promotion = Promotion(self.registry)

    def test_evaluate_gates_uses_configured_thresholds(self):
        # parsed_model_promotion_gates is a read-only computed property derived from
        # the raw model_promotion_gates JSON-string field -- patch the field, not the
        # property (it has no setter).
        from core import settings

        original = settings.model_promotion_gates
        try:
            settings.model_promotion_gates = '{"malicious_recall": {"operator": ">=", "threshold": 0.9}}'
            passing = self.promotion._evaluate_gates({"malicious_recall": 0.95}, source="candidate")
            failing = self.promotion._evaluate_gates({"malicious_recall": 0.5}, source="candidate")
        finally:
            settings.model_promotion_gates = original

        self.assertEqual(len(passing), 1)
        self.assertIsInstance(passing[0], GateResult)
        self.assertTrue(passing[0].passed)
        self.assertFalse(failing[0].passed)

    def test_evaluate_gates_missing_metric_fails_closed(self):
        from core import settings

        original = settings.model_promotion_gates
        try:
            settings.model_promotion_gates = '{"malicious_recall": {"operator": ">=", "threshold": 0.9}}'
            results = self.promotion._evaluate_gates({}, source="candidate")
        finally:
            settings.model_promotion_gates = original

        self.assertFalse(results[0].passed)
        self.assertIsNone(results[0].actual)

    def test_compare_to_champion_skipped_when_no_champion(self):
        from core import settings

        original = settings.promotion_require_champion_comparison
        try:
            settings.promotion_require_champion_comparison = True
            results = self.promotion._compare_to_champion({"malicious_recall": 0.9}, champion=None)
        finally:
            settings.promotion_require_champion_comparison = original
        self.assertEqual(results, [])

    def test_compare_to_champion_skipped_when_not_required(self):
        from core import settings

        original = settings.promotion_require_champion_comparison
        try:
            settings.promotion_require_champion_comparison = False
            champion = ModelReference(
                name="m", version="1", alias="champion", run_id="run-1", source="src", tags={}
            )
            results = self.promotion._compare_to_champion({"malicious_recall": 0.9}, champion=champion)
        finally:
            settings.promotion_require_champion_comparison = original
        self.assertEqual(results, [])

    def test_validate_feature_schema_rejects_missing_keys(self):
        with self.assertRaises(ModelValidationError):
            self.promotion._validate_feature_schema({"schema_version": "1.0.0"})  # missing "features"

    def test_validate_dataset_metadata_rejects_missing_keys(self):
        with self.assertRaises(ModelValidationError):
            self.promotion._validate_dataset_metadata({"dataset_version": "1.0.0"})


if __name__ == "__main__":
    unittest.main()
