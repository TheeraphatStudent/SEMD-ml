from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from core import features_config, settings
from ml.ml_pipeline import MLPipeline
from semd_ml.features.schema import FeatureSchema, build_feature_schema

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None
    MlflowException = Exception


class ModelRegistryError(RuntimeError):
    pass


class ModelValidationError(ModelRegistryError):
    pass


@dataclass(frozen=True)
class GateResult:
    metric: str
    operator: str
    threshold: float
    actual: Optional[float]
    passed: bool
    source: str


@dataclass(frozen=True)
class ModelReference:
    name: str
    version: str
    alias: Optional[str]
    run_id: str
    source: str
    tags: Dict[str, str]


class ModelRegistryManager:
    def __init__(
        self,
        client: Any | None = None,
        pipeline_factory: Callable[[], MLPipeline] = MLPipeline,
        artifact_downloader: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.model_name = settings.mlflow_registered_model_name
        self.candidate_alias = settings.mlflow_alias_candidate
        self.champion_alias = settings.mlflow_alias_champion
        self.previous_champion_alias = settings.mlflow_alias_previous_champion
        self.pipeline_factory = pipeline_factory
        self._artifact_downloader = artifact_downloader or self._default_download_artifact
        self.client = client
        self.last_error: Optional[str] = None

        if self.client is None and MlflowClient is not None and settings.mlflow_tracking_uri:
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            except Exception as exc:  # pragma: no cover - external dependency
                self.last_error = str(exc)
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def register_candidate(self, run_id: str) -> Dict[str, Any]:
        self._require_registry()
        self._ensure_registered_model()
        run = self.client.get_run(run_id)
        source = self._resolve_model_source(run_id)
        version = self.client.create_model_version(
            name=self.model_name,
            source=source,
            run_id=run_id,
            tags={
                "registered_at": self._utc_now(),
                "registered_model_name": self.model_name,
                "feature_schema_version": run.data.params.get("feature_schema_version", ""),
                "dataset_version": run.data.params.get("dataset_version", ""),
                "dataset_hash": run.data.params.get("dataset_hash", ""),
            },
        )
        self.client.set_registered_model_alias(self.model_name, self.candidate_alias, version.version)
        return {
            "model_name": self.model_name,
            "model_version": str(version.version),
            "run_id": run_id,
            "model_alias": self.candidate_alias,
            "source": source,
        }

    def validate_candidate(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        reference = self._get_reference(alias=self.candidate_alias if model_version is None else None, version=model_version)
        validation = self._validate_reference(reference)
        validation["model_alias"] = reference.alias
        validation["model_version"] = reference.version
        validation["model_name"] = reference.name
        return validation

    def promote_candidate(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        reference = self._get_reference(alias=self.candidate_alias if model_version is None else None, version=model_version)
        validation = self._validate_reference(reference)
        if not validation["passed"]:
            raise ModelValidationError("Candidate validation failed")

        previous = self._safe_get_reference(alias=self.champion_alias)
        if previous is not None:
            self.client.set_registered_model_alias(self.model_name, self.previous_champion_alias, previous.version)
            self.client.set_model_version_tag(
                self.model_name,
                previous.version,
                "replaced_at",
                self._utc_now(),
            )

        self.client.set_registered_model_alias(self.model_name, self.champion_alias, reference.version)
        metadata = {
            "promoted_at": self._utc_now(),
            "promotion_status": "approved",
            "promoted_from_alias": reference.alias or "explicit-version",
            "previous_champion_version": previous.version if previous is not None else "",
            "validation_summary": json.dumps(
                {
                    "gates_passed": validation["gates_passed"],
                    "champion_comparison_passed": validation["champion_comparison_passed"],
                    "smoke_tests_passed": validation["smoke_tests_passed"],
                },
                sort_keys=True,
            ),
        }
        for key, value in metadata.items():
            self.client.set_model_version_tag(self.model_name, reference.version, key, value)

        return {
            "model_name": self.model_name,
            "model_version": reference.version,
            "model_alias": self.champion_alias,
            "previous_champion_version": previous.version if previous is not None else None,
            "validation": validation,
        }

    def rollback_to_previous_champion(self) -> Dict[str, Any]:
        champion = self._safe_get_reference(alias=self.champion_alias)
        previous = self._safe_get_reference(alias=self.previous_champion_alias)
        if previous is None:
            raise ModelRegistryError("Rollback aborted: no previous champion alias is configured")

        self.client.set_registered_model_alias(self.model_name, self.champion_alias, previous.version)
        if champion is not None:
            self.client.set_registered_model_alias(self.model_name, self.previous_champion_alias, champion.version)
            self.client.set_model_version_tag(self.model_name, champion.version, "rolled_back_at", self._utc_now())

        self.client.set_model_version_tag(self.model_name, previous.version, "rollback_promoted_at", self._utc_now())
        return {
            "model_name": self.model_name,
            "champion_version": previous.version,
            "previous_champion_version": champion.version if champion is not None else None,
            "model_alias": self.champion_alias,
        }

    def load_reference(self, alias: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Any]:
        reference = self._get_reference(alias=alias, version=version)
        pipeline = self.pipeline_factory()
        artifact_path = self._artifact_downloader(reference.source)
        pipeline.load_artifact(artifact_path)

        expected_schema = build_feature_schema(features_config)
        if expected_schema.schema_version != pipeline.feature_schema.schema_version:
            raise ModelValidationError(
                "Loaded model feature schema version does not match runtime schema: "
                f"runtime={expected_schema.schema_version}, loaded={pipeline.feature_schema.schema_version}"
            )

        return {"reference": reference, "pipeline": pipeline}

    def _validate_reference(self, reference: ModelReference) -> Dict[str, Any]:
        run = self.client.get_run(reference.run_id)
        feature_schema = self._load_json_artifact(reference.run_id, "artifacts/feature_schema.json")
        dataset_metadata = self._load_json_artifact(reference.run_id, "artifacts/dataset_metadata.json")
        sample_predictions = self._load_json_artifact(reference.run_id, "artifacts/sample_predictions.json")

        self._validate_feature_schema(feature_schema)
        self._validate_dataset_metadata(dataset_metadata)

        gate_results = self._evaluate_gates(run.data.metrics, source="candidate")
        champion = self._safe_get_reference(alias=self.champion_alias)
        champion_results = self._compare_to_champion(run.data.metrics, champion)
        smoke_test_results = self._run_smoke_tests(reference, sample_predictions)

        passed = all(result.passed for result in gate_results)
        comparison_passed = all(result.passed for result in champion_results)
        smoke_passed = all(item["passed"] for item in smoke_test_results)
        return {
            "passed": passed and comparison_passed and smoke_passed,
            "feature_schema_version": feature_schema["schema_version"],
            "dataset_version": dataset_metadata["dataset_version"],
            "dataset_hash": dataset_metadata["dataset_hash"],
            "gate_results": [result.__dict__ for result in gate_results],
            "gates_passed": passed,
            "champion_comparison": [result.__dict__ for result in champion_results],
            "champion_comparison_passed": comparison_passed,
            "smoke_tests": smoke_test_results,
            "smoke_tests_passed": smoke_passed,
            "current_champion_version": champion.version if champion is not None else None,
        }

    def _compare_to_champion(self, candidate_metrics: Mapping[str, Any], champion: Optional[ModelReference]) -> List[GateResult]:
        if champion is None or not settings.promotion_require_champion_comparison:
            return []

        run = self.client.get_run(champion.run_id)
        champion_metrics = run.data.metrics
        results: List[GateResult] = []
        for metric, rule in settings.parsed_model_promotion_gates.items():
            candidate_value = self._coerce_metric(candidate_metrics.get(metric))
            champion_value = self._coerce_metric(champion_metrics.get(metric))
            if candidate_value is None or champion_value is None:
                results.append(
                    GateResult(
                        metric=metric,
                        operator="compare",
                        threshold=champion_value or 0.0,
                        actual=candidate_value,
                        passed=False,
                        source="champion",
                    )
                )
                continue

            operator = str(rule["operator"])
            passed = candidate_value >= champion_value if operator == ">=" else candidate_value <= champion_value
            results.append(
                GateResult(
                    metric=metric,
                    operator=operator,
                    threshold=champion_value,
                    actual=candidate_value,
                    passed=passed,
                    source="champion",
                )
            )
        return results

    def _run_smoke_tests(self, reference: ModelReference, sample_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        loader = self.load_reference(version=reference.version)
        pipeline: MLPipeline = loader["pipeline"]
        urls = settings.parsed_promotion_smoke_test_urls
        if sample_predictions:
            urls = [item["url"] for item in sample_predictions[:2] if item.get("url")] + urls

        results = []
        for url in urls[:4]:
            prediction = pipeline.predict(url)
            results.append(
                {
                    "url": url,
                    "passed": prediction["prediction"] in {"benign", "malicious"},
                    "model_version": reference.version,
                    "prediction": prediction["prediction"],
                }
            )
        return results

    def _validate_feature_schema(self, payload: Dict[str, Any]) -> None:
        if "schema_version" not in payload or "features" not in payload:
            raise ModelValidationError("Candidate artifact is missing feature schema metadata")
        runtime_schema = build_feature_schema(features_config)
        artifact_schema = FeatureSchema.from_dict(payload)
        if artifact_schema.schema_version != runtime_schema.schema_version:
            raise ModelValidationError(
                "Candidate feature schema mismatch: "
                f"runtime={runtime_schema.schema_version}, candidate={artifact_schema.schema_version}"
            )

    def _validate_dataset_metadata(self, payload: Dict[str, Any]) -> None:
        required = {"dataset_version", "dataset_hash", "total_records", "source_references"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ModelValidationError(f"Candidate dataset metadata is missing required keys: {', '.join(missing)}")

    def _evaluate_gates(self, metrics: Mapping[str, Any], source: str) -> List[GateResult]:
        results: List[GateResult] = []
        for metric, rule in settings.parsed_model_promotion_gates.items():
            actual = self._coerce_metric(metrics.get(metric))
            operator = str(rule["operator"])
            threshold = float(rule["threshold"])
            passed = False
            if actual is not None:
                passed = actual >= threshold if operator == ">=" else actual <= threshold
            results.append(
                GateResult(
                    metric=metric,
                    operator=operator,
                    threshold=threshold,
                    actual=actual,
                    passed=passed,
                    source=source,
                )
            )
        return results

    def _resolve_model_source(self, run_id: str) -> str:
        artifacts = self.client.list_artifacts(run_id, "artifacts")
        joblib_artifacts = [artifact.path for artifact in artifacts if artifact.path.endswith(".joblib")]
        if not joblib_artifacts:
            raise ModelRegistryError("No joblib model artifact was logged for the supplied run")
        return f"runs:/{run_id}/{joblib_artifacts[0]}"

    def _safe_get_reference(self, alias: str) -> Optional[ModelReference]:
        try:
            return self._get_reference(alias=alias)
        except ModelRegistryError:
            return None

    def _get_reference(self, alias: Optional[str] = None, version: Optional[str] = None) -> ModelReference:
        self._require_registry()
        if alias:
            try:
                model_version = self.client.get_model_version_by_alias(self.model_name, alias)
            except Exception as exc:
                raise ModelRegistryError(f"Alias '{alias}' is not assigned for model '{self.model_name}'") from exc
            return self._to_reference(model_version, alias=alias)
        if version:
            model_version = self.client.get_model_version(self.model_name, str(version))
            return self._to_reference(model_version, alias=None)
        raise ModelRegistryError("A model alias or explicit model version is required")

    def _to_reference(self, model_version: Any, alias: Optional[str]) -> ModelReference:
        tags = dict(getattr(model_version, "tags", {}) or {})
        return ModelReference(
            name=model_version.name,
            version=str(model_version.version),
            alias=alias,
            run_id=model_version.run_id,
            source=model_version.source,
            tags={str(key): str(value) for key, value in tags.items()},
        )

    def _load_json_artifact(self, run_id: str, artifact_path: str) -> Any:
        artifact = self.client.download_artifacts(run_id, artifact_path)
        return json.loads(Path(artifact).read_text(encoding="utf-8"))

    def _default_download_artifact(self, artifact_uri: str) -> str:
        if mlflow is None:  # pragma: no cover - guarded by _require_registry
            raise ModelRegistryError("mlflow is not installed")
        return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)

    def _ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.model_name)
        except Exception:
            self.client.create_registered_model(self.model_name)

    def _require_registry(self) -> None:
        if not self.available:
            raise ModelRegistryError("MLflow model registry is unavailable")

    def _coerce_metric(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class CachedChampionModelLoader:
    def __init__(
        self,
        registry_manager: Optional[ModelRegistryManager] = None,
        pipeline_factory: Callable[[], MLPipeline] = MLPipeline,
    ) -> None:
        self.registry_manager = registry_manager or ModelRegistryManager(pipeline_factory=pipeline_factory)
        self.pipeline_factory = pipeline_factory
        self._cached_pipeline: Optional[MLPipeline] = None
        self._cached_reference: Optional[ModelReference] = None

    def load(self, selector: Optional[str] = None) -> Dict[str, Any]:
        alias = None
        version = None
        if selector in (None, "", "latest", settings.mlflow_alias_champion):
            alias = settings.mlflow_alias_champion
        elif str(selector).isdigit():
            version = str(selector)
        else:
            alias = str(selector)

        if alias == settings.mlflow_alias_champion and self._cached_pipeline is not None and self._cached_reference is not None:
            return {"pipeline": self._cached_pipeline, "reference": self._cached_reference}

        try:
            loaded = self.registry_manager.load_reference(alias=alias, version=version)
        except Exception as exc:
            if not settings.mlflow_local_fallback_enabled:
                raise ModelRegistryError(f"Unable to load model from MLflow registry: {exc}") from exc
            loaded = self._load_local_fallback(exc)

        if alias == settings.mlflow_alias_champion:
            self._cached_pipeline = loaded["pipeline"]
            self._cached_reference = loaded["reference"]
        return loaded

    def clear_cache(self) -> None:
        self._cached_pipeline = None
        self._cached_reference = None

    def predict(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        loaded = self.load(selector=selector)
        pipeline: MLPipeline = loaded["pipeline"]
        reference: ModelReference = loaded["reference"]
        prediction = pipeline.predict(url)
        prediction.update(
            {
                "model_name": reference.name,
                "model_version": reference.version,
                "model_alias": reference.alias or ("version" if selector else settings.mlflow_alias_champion),
            }
        )
        return prediction

    def _load_local_fallback(self, original_error: Exception) -> Dict[str, Any]:
        path = settings.mlflow_local_fallback_model_path
        version = settings.mlflow_local_fallback_model_version
        name = settings.mlflow_local_fallback_model_name or settings.mlflow_registered_model_name
        if not path or not version:
            raise ModelRegistryError(
                "MLflow registry is unavailable and local fallback is not fully configured"
            ) from original_error

        pipeline = self.pipeline_factory()
        pipeline.load_artifact(path)
        reference = ModelReference(
            name=name,
            version=version,
            alias="local-fallback",
            run_id="local-fallback",
            source=path,
            tags={"fallback_reason": str(original_error)},
        )
        return {"pipeline": pipeline, "reference": reference}
