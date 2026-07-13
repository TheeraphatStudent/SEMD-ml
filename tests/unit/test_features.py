import unittest

from core import features_config
from semd_ml.features.extractor import URLFeatureExtractor
from semd_ml.features.schema import build_feature_schema
from semd_ml.features.url_normalizer import normalize_url


class URLNormalizationTests(unittest.TestCase):
    def test_normalization_is_deterministic(self):
        result = normalize_url("  Example.com:80/path?a=1&b=2#frag  ")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.normalized_url, "http://example.com/path?a=1&b=2")

    def test_unicode_domain_is_punycoded(self):
        result = normalize_url("https://аррӏе.com/login")
        self.assertTrue(result.is_valid)
        self.assertIn("xn--", result.normalized_url)

    def test_invalid_url_is_reported(self):
        result = normalize_url("not a url")
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.error)


class FeatureExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = URLFeatureExtractor(features_config)
        cls.schema = build_feature_schema(features_config)

    def test_feature_order_matches_schema(self):
        features = self.extractor.extract("https://example.com")
        self.assertEqual(list(features.keys()), self.schema.feature_names)

    def test_ip_detection(self):
        features = self.extractor.extract("http://192.168.0.1/login")
        self.assertEqual(features["ip_address_flag"], 1.0)

    def test_punycode_detection(self):
        features = self.extractor.extract("https://аррӏе.com")
        self.assertEqual(features["punycode_domain_flag"], 1.0)
        self.assertEqual(features["unicode_domain_flag"], 1.0)

    def test_port_handling(self):
        safe = self.extractor.extract("http://example.com:80/path")
        risky = self.extractor.extract("http://example.com:8080/path")
        self.assertEqual(safe["port_in_url_flag"], 1.0)
        self.assertEqual(safe["non_standard_port_flag"], 0.0)
        self.assertEqual(risky["port_in_url_flag"], 1.0)
        self.assertEqual(risky["non_standard_port_flag"], 1.0)

    def test_suspicious_extension_detection(self):
        features = self.extractor.extract("https://example.com/dropper.EXE")
        self.assertEqual(features["suspicious_extension_flag"], 1.0)
        self.assertEqual(features["executable_extension_flag"], 1.0)

    def test_invalid_url_extracts_schema_aligned_defaults(self):
        features = self.extractor.extract("%%%%")
        self.assertEqual(list(features.keys()), self.schema.feature_names)
        self.assertEqual(features["domain_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
