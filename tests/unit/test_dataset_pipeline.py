import unittest

import pandas as pd

from data.dataset_pipeline import DatasetPipeline
from semd_ml.data.splitters import DatasetSplitter
from semd_ml.data.validators import DatasetValidator
from semd_ml.data.versioning import compute_dataset_hash


class DatasetValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = DatasetPipeline()
        cls.validator = DatasetValidator(cls.pipeline.data_dict)

    def _frame(self, rows):
        frame = pd.DataFrame(rows)
        frame["source"] = "test.csv"
        frame["source_row"] = range(1, len(frame) + 1)
        frame["source_url_column"] = "url"
        frame["source_label_column"] = "raw_label"
        frame["type"] = None
        return frame

    def test_duplicate_detection(self):
        frame = self._frame(
            [
                {"url": "example.com", "raw_label": "benign"},
                {"url": "http://example.com", "raw_label": "benign"},
            ]
        )
        result = self.validator.validate(frame)
        self.assertEqual(result.stats["duplicate_count"], 1)

    def test_conflicting_labels(self):
        frame = self._frame(
            [
                {"url": "example.com", "raw_label": "benign"},
                {"url": "http://example.com", "raw_label": "malicious"},
            ]
        )
        result = self.validator.validate(frame)
        self.assertIn("Conflicting labels: 1", result.errors)

    def test_invalid_url(self):
        frame = self._frame([{"url": "not a url", "raw_label": "benign"}])
        result = self.validator.validate(frame)
        self.assertEqual(result.stats["invalid_url_count"], 1)

    def test_dataset_cleanup_drops_invalid_duplicate_and_conflicting_rows(self):
        frame = self._frame(
            [
                {"url": "https://good.example.com", "raw_label": "benign"},
                {"url": "not a url", "raw_label": "benign"},
                {"url": "dup.example.com", "raw_label": "malicious"},
                {"url": "http://dup.example.com", "raw_label": "malicious"},
                {"url": "conflict.example.com", "raw_label": "benign"},
                {"url": "http://conflict.example.com", "raw_label": "malicious"},
            ]
        )
        cleaned, validation = self.validator.clean(frame)

        self.assertEqual(
            sorted(cleaned["normalized_url"]),
            ["http://dup.example.com", "https://good.example.com"],
        )
        self.assertEqual(validation.stats["invalid_url_count"], 1)
        # Both the same-label pair and the conflicting-label pair normalize to a
        # repeated URL, so validate() counts two duplicate pairs pre-conflict-filtering.
        self.assertEqual(validation.stats["duplicate_count"], 2)
        self.assertIn("Conflicting labels: 1", validation.errors)

    def test_hash_is_stable(self):
        cleaned_a = pd.DataFrame(
            [
                {"normalized_url": "http://a.com", "label": "benign", "source": "a.csv"},
                {"normalized_url": "http://b.com", "label": "malicious", "source": "a.csv"},
            ]
        )
        cleaned_b = cleaned_a.iloc[::-1].reset_index(drop=True)
        self.assertEqual(compute_dataset_hash(cleaned_a), compute_dataset_hash(cleaned_b))


class DatasetSplitTests(unittest.TestCase):
    def test_split_avoids_registered_domain_leakage(self):
        splitter = DatasetSplitter(random_state=42, test_size=0.3)
        X = pd.DataFrame({"f1": range(8)})
        y = pd.Series(["benign", "benign", "malicious", "malicious", "benign", "benign", "malicious", "malicious"])
        groups = pd.Series(
            [
                "a.com",
                "a.com",
                "b.com",
                "b.com",
                "c.com",
                "c.com",
                "d.com",
                "d.com",
            ]
        )
        split = splitter.split(X, y, groups)
        train_groups = set(groups.iloc[split.train_indices].tolist())
        test_groups = set(groups.iloc[split.test_indices].tolist())
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_pipeline_feature_order_is_deterministic(self):
        pipeline = DatasetPipeline()
        cleaned = pd.DataFrame(
            [
                {
                    "url": "http://example.com",
                    "label": "benign",
                    "normalized_url": "http://example.com",
                    "registered_domain": "example.com",
                    "source": "test.csv",
                    "source_row": 1,
                }
            ]
        )
        X, _ = pipeline.extract_features(cleaned)
        self.assertEqual(X.columns.tolist(), pipeline.feature_schema.feature_names)


if __name__ == "__main__":
    unittest.main()
