"""T06 (docs/refactoring-plan.md): dedicated coverage for the algorithm factory.

Previously only exercised indirectly through test_training_pipeline.py's full
train_models() runs. This locks ModelFactory's own contract independently of
the rest of the training pipeline.
"""

from __future__ import annotations

import unittest

from ml.model_factory import ModelFactory


class ModelFactoryTests(unittest.TestCase):
    def setUp(self):
        self.factory = ModelFactory()

    def test_identifiers_include_core_algorithms(self):
        identifiers = self.factory.identifiers()
        self.assertIn("svm", identifiers)
        self.assertIn("random_forest", identifiers)
        self.assertIn("gradient_boosting", identifiers)

    def test_build_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            self.factory.build("not-a-real-algorithm")

    def test_build_applies_configured_random_state(self):
        estimator = self.factory.build("random_forest")
        self.assertEqual(estimator.random_state, self.factory.random_state)

    def test_build_overrides_take_precedence_over_defaults(self):
        estimator = self.factory.build("random_forest", overrides={"n_estimators": 7})
        self.assertEqual(estimator.n_estimators, 7)

    def test_available_models_returns_definition_per_identifier(self):
        definitions = self.factory.available_models()
        for identifier in self.factory.identifiers():
            self.assertIn(identifier, definitions)
            self.assertEqual(definitions[identifier].identifier, identifier)


if __name__ == "__main__":
    unittest.main()
