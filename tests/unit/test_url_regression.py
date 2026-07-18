import unittest

from core import features_config
from features.extractor import URLFeatureExtractor
from features.schema import build_feature_schema
from features.url_normalizer import normalize_url


class RegressionURLTests(unittest.TestCase):
    """Table-driven regression suite covering the canonical URL shapes the
    normalizer and feature extractor must handle without raising."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = URLFeatureExtractor(features_config)
        cls.schema = build_feature_schema(features_config)

    def _assert_schema_aligned(self, url):
        features = self.extractor.extract(url)
        self.assertEqual(list(features.keys()), self.schema.feature_names)
        return features

    def test_normal_https_url(self):
        result = normalize_url("https://www.example.com/home")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.scheme, "https")
        features = self._assert_schema_aligned("https://www.example.com/home")
        self.assertEqual(features["http_token"], 0.0)

    def test_http_url(self):
        result = normalize_url("http://www.example.com/home")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.scheme, "http")
        features = self._assert_schema_aligned("http://www.example.com/home")
        self.assertEqual(features["http_token"], 1.0)

    def test_ip_based_url(self):
        result = normalize_url("http://192.168.1.10/admin")
        self.assertTrue(result.is_valid)
        features = self._assert_schema_aligned("http://192.168.1.10/admin")
        self.assertEqual(features["ip_address_flag"], 1.0)

    def test_punycode_url(self):
        result = normalize_url("https://xn--80ak6aa92e.com/login")
        self.assertTrue(result.is_valid)
        features = self._assert_schema_aligned("https://xn--80ak6aa92e.com/login")
        self.assertEqual(features["punycode_domain_flag"], 1.0)

    def test_suspicious_file_extension_url(self):
        result = normalize_url("https://example.com/invoice.exe")
        self.assertTrue(result.is_valid)
        features = self._assert_schema_aligned("https://example.com/invoice.exe")
        self.assertEqual(features["suspicious_extension_flag"], 1.0)
        self.assertEqual(features["executable_extension_flag"], 1.0)

    def test_long_phishing_style_url(self):
        url = (
            "http://secure-login-account-verification-update.bad-example.net"
            "/wp-admin/verify/account/update/session?user=victim&token=" + "a" * 80
        )
        result = normalize_url(url)
        self.assertTrue(result.is_valid)
        features = self._assert_schema_aligned(url)
        self.assertEqual(features["long_url_length"], 1.0)

    def test_url_with_explicit_port(self):
        result = normalize_url("http://example.com:8080/path")
        self.assertTrue(result.is_valid)
        self.assertTrue(result.had_explicit_port)
        features = self._assert_schema_aligned("http://example.com:8080/path")
        self.assertEqual(features["port_in_url_flag"], 1.0)
        self.assertEqual(features["non_standard_port_flag"], 1.0)

    def test_malformed_url(self):
        result = normalize_url("http://[not-valid")
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.error)
        features = self._assert_schema_aligned("http://[not-valid")
        self.assertEqual(features["domain_length"], 0.0)

    def test_empty_url(self):
        result = normalize_url("")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error, "empty_url")
        features = self._assert_schema_aligned("")
        self.assertEqual(features["domain_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
