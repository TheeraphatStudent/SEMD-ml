import unittest

from core.config import MLServiceSettings


class MLflowSettingsTests(unittest.TestCase):
    def setUp(self):
        self.settings = MLServiceSettings()

    def test_default_promotion_gates_parse_to_expected_metrics(self):
        gates = self.settings.parsed_model_promotion_gates
        self.assertEqual(
            set(gates.keys()),
            {"malicious_recall", "malicious_f1", "false_negative_rate", "prediction_latency_ms"},
        )
        self.assertEqual(gates["malicious_recall"], {"operator": ">=", "threshold": 0.95})
        self.assertEqual(gates["false_negative_rate"], {"operator": "<=", "threshold": 0.05})

    def test_default_smoke_test_urls_parse_to_list_of_strings(self):
        urls = self.settings.parsed_promotion_smoke_test_urls
        self.assertIsInstance(urls, list)
        self.assertTrue(all(isinstance(url, str) for url in urls))
        self.assertIn("https://example.com", urls)

    def test_invalid_promotion_gates_json_raises(self):
        settings = MLServiceSettings(model_promotion_gates="not-json")
        with self.assertRaises(ValueError):
            _ = settings.parsed_model_promotion_gates

    def test_promotion_gates_must_decode_to_object(self):
        settings = MLServiceSettings(model_promotion_gates="[1, 2, 3]")
        with self.assertRaises(ValueError):
            _ = settings.parsed_model_promotion_gates

    def test_invalid_smoke_test_urls_json_raises(self):
        settings = MLServiceSettings(promotion_smoke_test_urls="not-json")
        with self.assertRaises(ValueError):
            _ = settings.parsed_promotion_smoke_test_urls

    def test_smoke_test_urls_must_decode_to_list_of_strings(self):
        settings = MLServiceSettings(promotion_smoke_test_urls='["https://example.com", 1]')
        with self.assertRaises(ValueError):
            _ = settings.parsed_promotion_smoke_test_urls

    def test_registry_aliases_have_expected_defaults(self):
        self.assertEqual(self.settings.mlflow_alias_candidate, "candidate")
        self.assertEqual(self.settings.mlflow_alias_champion, "champion")
        self.assertEqual(self.settings.mlflow_alias_previous_champion, "previous-champion")


if __name__ == "__main__":
    unittest.main()
