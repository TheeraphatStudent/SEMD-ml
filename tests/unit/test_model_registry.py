from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any
import unittest

from data.dataset_pipeline import DatasetPipeline
from core import settings
from ml.ml_pipeline import MLPipeline
from tracking.model_registry import (
    CachedChampionModelLoader,
    ModelRegistryError,
    ModelRegistryManager,
    ModelValidationError,
)


def write_fixture_dataset(root: Path, name: str = "fixture.csv", benign: int = 12, malicious: int = 8) -> str:
    rows = []
    for idx in range(benign):
        rows.append({"url": f"https://benign{idx}.example.com/home", "label": "benign"})
    for idx in range(malicious):
        rows.append({"url": f"http://secure-login{idx}.bad-example.net/verify?token={idx}", "label": "malicious"})
    import pandas as pd

    pd.DataFrame(rows).to_csv(root / name, index=False)
    return name


class FakeModelVersion:
    def __init__(self, name: str, version: str, run_id: str, source: str, tags: dict[str, str] | None = None):
        self.name = name
        self.version = version
        self.run_id = run_id
        self.source = source
        self.tags = tags or {}


class FakeMlflowClient:
    def __init__(self, runs: dict[str, Any], artifact_dirs: dict[str, Path]):
        self.runs = runs
        self.artifact_dirs = artifact_dirs
        self.aliases: dict[str, str] = {}
        self.versions: dict[str, FakeModelVersion] = {}
        self.version_counter = 0
        self.version_tags: dict[str, dict[str, str]] = {}
        self.model_exists = False

    def get_run(self, run_id: str):
        return self.runs[run_id]

    def get_registered_model(self, _name: str):
        if not self.model_exists:
            raise KeyError("missing")
        return SimpleNamespace(name=_name)

    def create_registered_model(self, name: str):
        self.model_exists = True
        return SimpleNamespace(name=name)

    def list_artifacts(self, run_id: str, path: str):
        root = self.artifact_dirs[run_id] / path
        return [SimpleNamespace(path=f"{path}/{item.name}") for item in root.iterdir()]

    def create_model_version(self, name: str, source: str, run_id: str, tags: dict[str, str]):
        self.version_counter += 1
        version = str(self.version_counter)
        self.versions[version] = FakeModelVersion(name=name, version=version, run_id=run_id, source=source, tags=tags)
        self.version_tags[version] = dict(tags)
        return self.versions[version]

    def set_registered_model_alias(self, _name: str, alias: str, version: str):
        self.aliases[alias] = str(version)

    def get_model_version_by_alias(self, _name: str, alias: str):
        version = self.aliases[alias]
        model_version = self.versions[version]
        model_version.tags = self.version_tags.get(version, {})
        return model_version

    def get_model_version(self, _name: str, version: str):
        model_version = self.versions[str(version)]
        model_version.tags = self.version_tags.get(str(version), {})
        return model_version

    def download_artifacts(self, run_id: str, artifact_path: str):
        return str(self.artifact_dirs[run_id] / artifact_path)

    def set_model_version_tag(self, _name: str, version: str, key: str, value: str):
        self.version_tags.setdefault(str(version), {})[key] = value


class ModelRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_name = write_fixture_dataset(self.root)

        self.dataset_pipeline = DatasetPipeline()
        self.dataset_pipeline.dataset_path = str(self.root)
        self.dataset_pipeline.extraction_path = str(self.root / "extraction")
        self.ml_pipeline = MLPipeline()
        self.ml_pipeline.models_path = str(self.root / "models")
        self.dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        self.training = self.ml_pipeline.train_models(self.dataset, ["random_forest"], run_id="registry-run")
        self.artifact_path = Path(self.training["best_artifact_path"])

        self.original_fallback = {
            "enabled": settings.mlflow_local_fallback_enabled,
            "path": settings.mlflow_local_fallback_model_path,
            "version": settings.mlflow_local_fallback_model_version,
            "name": settings.mlflow_local_fallback_model_name,
        }

    def tearDown(self) -> None:
        settings.mlflow_local_fallback_enabled = self.original_fallback["enabled"]
        settings.mlflow_local_fallback_model_path = self.original_fallback["path"]
        settings.mlflow_local_fallback_model_version = self.original_fallback["version"]
        settings.mlflow_local_fallback_model_name = self.original_fallback["name"]
        self.temp_dir.cleanup()

    def test_register_promote_and_rollback_workflow(self):
        candidate_artifacts = self._create_run_artifacts(
            "run-candidate",
            metrics={
                "malicious_recall": 0.98,
                "malicious_f1": 0.96,
                "false_negative_rate": 0.02,
                "prediction_latency_ms": 50.0,
            },
        )
        champion_artifacts = self._create_run_artifacts(
            "run-champion",
            metrics={
                "malicious_recall": 0.95,
                "malicious_f1": 0.94,
                "false_negative_rate": 0.03,
                "prediction_latency_ms": 55.0,
            },
        )
        client = FakeMlflowClient(
            runs={candidate_artifacts["run_id"]: candidate_artifacts["run"], champion_artifacts["run_id"]: champion_artifacts["run"]},
            artifact_dirs={
                candidate_artifacts["run_id"]: candidate_artifacts["root"],
                champion_artifacts["run_id"]: champion_artifacts["root"],
            },
        )
        manager = self._build_manager(client)

        champion_version = manager.register_candidate(champion_artifacts["run_id"])["model_version"]
        manager.promote_candidate(champion_version)

        candidate_registration = manager.register_candidate(candidate_artifacts["run_id"])
        self.assertEqual(candidate_registration["model_alias"], "candidate")
        validation = manager.validate_candidate()
        self.assertTrue(validation["passed"])

        promoted = manager.promote_candidate(candidate_registration["model_version"])
        self.assertEqual(promoted["model_alias"], "champion")
        self.assertEqual(client.aliases["champion"], candidate_registration["model_version"])
        self.assertEqual(client.aliases["previous-champion"], champion_version)

        rolled_back = manager.rollback_to_previous_champion()
        self.assertEqual(rolled_back["champion_version"], champion_version)
        self.assertEqual(client.aliases["champion"], champion_version)
        self.assertEqual(client.aliases["previous-champion"], candidate_registration["model_version"])

    def test_promotion_gates_reject_bad_candidate(self):
        bad_candidate = self._create_run_artifacts(
            "run-bad",
            metrics={
                "malicious_recall": 0.90,
                "malicious_f1": 0.90,
                "false_negative_rate": 0.10,
                "prediction_latency_ms": 250.0,
            },
        )
        client = FakeMlflowClient(
            runs={bad_candidate["run_id"]: bad_candidate["run"]},
            artifact_dirs={bad_candidate["run_id"]: bad_candidate["root"]},
        )
        manager = self._build_manager(client)
        manager.register_candidate(bad_candidate["run_id"])

        validation = manager.validate_candidate()
        self.assertFalse(validation["passed"])
        self.assertFalse(validation["gates_passed"])
        with self.assertRaises(ModelValidationError):
            manager.promote_candidate()

    def test_schema_mismatch_is_rejected(self):
        bad_schema = self._create_run_artifacts(
            "run-schema",
            metrics={
                "malicious_recall": 0.98,
                "malicious_f1": 0.96,
                "false_negative_rate": 0.02,
                "prediction_latency_ms": 50.0,
            },
            schema_version="999.0.0",
        )
        client = FakeMlflowClient(
            runs={bad_schema["run_id"]: bad_schema["run"]},
            artifact_dirs={bad_schema["run_id"]: bad_schema["root"]},
        )
        manager = self._build_manager(client)
        manager.register_candidate(bad_schema["run_id"])
        with self.assertRaises(ModelValidationError):
            manager.validate_candidate()

    def test_champion_loader_caches_model_and_supports_local_fallback(self):
        candidate = self._create_run_artifacts(
            "run-loader",
            metrics={
                "malicious_recall": 0.98,
                "malicious_f1": 0.96,
                "false_negative_rate": 0.02,
                "prediction_latency_ms": 50.0,
            },
        )
        client = FakeMlflowClient(
            runs={candidate["run_id"]: candidate["run"]},
            artifact_dirs={candidate["run_id"]: candidate["root"]},
        )
        manager = self._build_manager(client)
        version = manager.register_candidate(candidate["run_id"])["model_version"]
        manager.promote_candidate(version)

        loader = CachedChampionModelLoader(registry_manager=manager)
        first = loader.predict("https://example.com")
        second = loader.predict("https://example.org")
        self.assertEqual(first["model_alias"], "champion")
        self.assertEqual(second["model_version"], version)
        self.assertIsNotNone(loader._cached_pipeline)

        class FailingRegistry:
            def load_reference(self, *args, **kwargs):
                raise ModelRegistryError("registry offline")

        settings.mlflow_local_fallback_enabled = True
        settings.mlflow_local_fallback_model_path = str(self.artifact_path)
        settings.mlflow_local_fallback_model_version = "local-1"
        settings.mlflow_local_fallback_model_name = "semd-malicious-url-detector"

        fallback_loader = CachedChampionModelLoader(registry_manager=FailingRegistry())
        fallback = fallback_loader.predict("https://fallback.example.com")
        self.assertEqual(fallback["model_alias"], "local-fallback")
        self.assertEqual(fallback["model_version"], "local-1")

    def test_loader_raises_when_registry_is_unavailable_and_fallback_is_disabled(self):
        class FailingRegistry:
            def load_reference(self, *args, **kwargs):
                raise ModelRegistryError("registry offline")

        settings.mlflow_local_fallback_enabled = False
        loader = CachedChampionModelLoader(registry_manager=FailingRegistry())
        with self.assertRaises(ModelRegistryError):
            loader.predict("https://example.com")

    def _build_manager(self, client: FakeMlflowClient) -> ModelRegistryManager:
        def downloader(source: str) -> str:
            if source.startswith("runs:/"):
                _, rest = source.split("runs:/", 1)
                run_id, artifact_path = rest.split("/", 1)
                return str(client.artifact_dirs[run_id] / artifact_path)
            return source

        return ModelRegistryManager(client=client, artifact_downloader=downloader)

    def _create_run_artifacts(
        self,
        run_id: str,
        metrics: dict[str, float],
        schema_version: str | None = None,
    ) -> dict[str, Any]:
        root = self.root / run_id
        artifacts_dir = root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        feature_schema = dict(self.dataset["feature_schema"])
        if schema_version is not None:
            feature_schema["schema_version"] = schema_version
        (artifacts_dir / "feature_schema.json").write_text(json.dumps(feature_schema), encoding="utf-8")
        (artifacts_dir / "dataset_metadata.json").write_text(json.dumps(self.dataset["dataset_metadata"]), encoding="utf-8")
        (artifacts_dir / "sample_predictions.json").write_text(
            json.dumps(
                [
                    {"url": "https://example.com", "predicted_label": "benign"},
                    {"url": "http://secure-login.bad-example.net/verify", "predicted_label": "malicious"},
                ]
            ),
            encoding="utf-8",
        )
        model_path = artifacts_dir / self.artifact_path.name
        model_path.write_bytes(self.artifact_path.read_bytes())

        run = SimpleNamespace(
            info=SimpleNamespace(run_id=run_id),
            data=SimpleNamespace(
                metrics=metrics,
                params={
                    "feature_schema_version": feature_schema["schema_version"],
                    "dataset_version": self.dataset["dataset_metadata"]["dataset_version"],
                    "dataset_hash": self.dataset["dataset_metadata"]["dataset_hash"],
                },
            ),
        )
        return {"run_id": run_id, "run": run, "root": root}


if __name__ == "__main__":
    unittest.main()
