import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.loaders import UnsupportedDatasetFormatError, load_dataset_file
from data.repositories import DatasetRepository
from data.validators import DatasetValidator
from pipelines.dataset_build_pipeline import DatasetBuildPipeline

DATA_DICT = {
    "fields": {
        "url": ["url", "input", "target", "text"],
        "class": ["label", "class", "output", "type"],
    },
    "class_mapping": {
        "benign": [0, "benign", "legitimate", "normal"],
        "malicious": [1, 2, 3, "malicious", "malware", "phishing", "defacement", "redirect", "spam"],
    },
}


class LoadersTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reads_standard_csv(self):
        path = self.root / "data.csv"
        pd.DataFrame({"url": ["a.com"], "label": ["benign"]}).to_csv(path, index=False)
        frame = load_dataset_file(path)
        self.assertEqual(list(frame.columns), ["url", "label"])

    def test_falls_back_to_semicolon_separator(self):
        path = self.root / "data.csv"
        path.write_text("url;label\na.com;benign\nb.com;malicious\n", encoding="utf-8")
        frame = load_dataset_file(path)
        self.assertEqual(list(frame.columns), ["url", "label"])
        self.assertEqual(len(frame), 2)

    def test_unsupported_extension_raises(self):
        path = self.root / "data.txt"
        path.write_text("nothing", encoding="utf-8")
        with self.assertRaises(UnsupportedDatasetFormatError):
            load_dataset_file(path)


class DatasetRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "a.csv").write_text("url,label\na.com,benign\n", encoding="utf-8")
        (self.root / "b.xlsx").write_bytes(b"")
        (self.root / "merged.csv").write_text("url,label\n", encoding="utf-8")
        (self.root / "notes.txt").write_text("skip me", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_file_list_expands_raw_alias_and_skips_merged(self):
        repo = DatasetRepository(str(self.root))
        resolved = repo.resolve_file_list(["dataset/raw"])
        self.assertEqual(sorted(resolved), ["a.csv", "b.xlsx"])

    def test_resolve_file_list_passes_through_explicit_names(self):
        repo = DatasetRepository(str(self.root))
        resolved = repo.resolve_file_list(["a.csv"])
        self.assertEqual(resolved, ["a.csv"])

    def test_save_merged_writes_csv_and_metadata(self):
        repo = DatasetRepository(str(self.root))
        merged = pd.DataFrame({"url": ["a.com", "b.com"], "label": ["benign", "malicious"]})
        repo.save_merged(merged, [{"source": "a.csv", "records": 2}])

        saved = pd.read_csv(self.root / "merged.csv")
        self.assertEqual(len(saved), 2)
        metadata = json.loads((self.root / "merged.metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["total_records"], 2)
        self.assertEqual(metadata["source_references"][0]["source"], "a.csv")


class DatasetBuildPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.validator = DatasetValidator(DATA_DICT)
        self.pipeline = DatasetBuildPipeline(str(self.root), self.validator)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_merges_validates_cleans_and_versions(self):
        pd.DataFrame(
            {
                "url": ["https://good.example.com", "not a url", "https://good.example.com"],
                "label": ["benign", "benign", "benign"],
            }
        ).to_csv(self.root / "fixture.csv", index=False)

        result = self.pipeline.build(["fixture.csv"])

        self.assertEqual(len(result.cleaned), 1)
        self.assertIn("dataset_hash", result.dataset_metadata)
        self.assertEqual(result.dataset_metadata["benign_count"], 1)
        self.assertTrue((self.root / "merged.csv").exists())
        self.assertTrue((self.root / "merged.metadata.json").exists())

    def test_build_raises_when_nothing_loads(self):
        with self.assertRaises(ValueError):
            self.pipeline.build(["missing.csv"])


if __name__ == "__main__":
    unittest.main()
