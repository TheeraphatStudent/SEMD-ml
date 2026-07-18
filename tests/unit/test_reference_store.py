import unittest
from pathlib import Path

import pandas as pd

from features.reference_store import ReferenceStore


class ReferenceStoreTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_values_from_existing_csv(self):
        pd.DataFrame({"value": ["Foo", "Bar", "Bar"]}).to_csv(self.root / "brand_keyword.csv", index=False)
        store = ReferenceStore(self.root)

        result = store.load("brand_keyword", default=["fallback"])

        self.assertEqual(result, {"foo", "bar"})
        self.assertTrue(store.diagnostics["brand_keyword"].loaded)
        self.assertFalse(store.diagnostics["brand_keyword"].used_default)
        self.assertIsNone(store.diagnostics["brand_keyword"].error)
        self.assertFalse(store.has_failures())

    def test_missing_file_falls_back_to_default_and_records_diagnostic(self):
        store = ReferenceStore(self.root)

        result = store.load("suspicious_tld", default=["tk", "ml"])

        self.assertEqual(result, {"tk", "ml"})
        diagnostic = store.diagnostics["suspicious_tld"]
        self.assertTrue(diagnostic.used_default)
        self.assertIsNotNone(diagnostic.error)
        self.assertTrue(store.has_failures())

    def test_missing_value_column_falls_back_to_default(self):
        pd.DataFrame({"other": ["x"]}).to_csv(self.root / "free_hosting.csv", index=False)
        store = ReferenceStore(self.root)

        result = store.load("free_hosting", default=["freehost"])

        self.assertEqual(result, {"freehost"})
        self.assertTrue(store.diagnostics["free_hosting"].used_default)
        self.assertIn("missing 'value' column", store.diagnostics["free_hosting"].error)

    def test_transform_is_applied_to_loaded_and_default_values(self):
        pd.DataFrame({"value": ["*.XYZ"]}).to_csv(self.root / "suspicious_tld.csv", index=False)
        store = ReferenceStore(self.root)

        result = store.load(
            "suspicious_tld",
            default=["*.TOP"],
            transform=lambda value: value.lstrip("*.").lower(),
        )

        self.assertEqual(result, {"xyz"})


if __name__ == "__main__":
    unittest.main()
