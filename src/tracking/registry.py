from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from core import settings

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelReference:
    name: str
    version: str
    alias: Optional[str]
    run_id: str
    source: str
    tags: Dict[str, str]


class Registry:
    """Pure MLflow model-registry CRUD: register a run as a version, resolve
    alias/version/run_id references, rollback, and load a referenced artifact
    into a pipeline. No promotion-gate policy here -- see tracking/promotion.py,
    which composes a Registry instead of duplicating this.
    """

    def __init__(
        self,
        client: Any | None = None,
        pipeline_factory: Callable[[], Any] = None,
        artifact_downloader: Optional[Callable[[str], str]] = None,
    ) -> None:
        if pipeline_factory is None:
            from ml.ml_pipeline import MLPipeline

            pipeline_factory = MLPipeline
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

    def require_available(self) -> None:
        if not self.available:
            raise ModelRegistryError("MLflow model registry is unavailable")

    def register_candidate(self, run_id: str) -> Dict[str, Any]:
        self.require_available()
        self.ensure_registered_model()
        run = self.client.get_run(run_id)
        source = self.resolve_model_source(run_id)
        version = self.client.create_model_version(
            name=self.model_name,
            source=source,
            run_id=run_id,
            tags={
                "registered_at": self.utc_now(),
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

    def rollback_to_previous_champion(self) -> Dict[str, Any]:
        champion = self.safe_get_reference(alias=self.champion_alias)
        previous = self.safe_get_reference(alias=self.previous_champion_alias)
        if previous is None:
            raise ModelRegistryError("Rollback aborted: no previous champion alias is configured")

        self.client.set_registered_model_alias(self.model_name, self.champion_alias, previous.version)
        if champion is not None:
            self.client.set_registered_model_alias(self.model_name, self.previous_champion_alias, champion.version)
            self.client.set_model_version_tag(self.model_name, champion.version, "rolled_back_at", self.utc_now())

        self.client.set_model_version_tag(self.model_name, previous.version, "rollback_promoted_at", self.utc_now())
        return {
            "model_name": self.model_name,
            "champion_version": previous.version,
            "previous_champion_version": champion.version if champion is not None else None,
            "model_alias": self.champion_alias,
        }

    def load_reference(self, alias: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Any]:
        from core import features_config
        from features.schema import build_feature_schema

        reference = self.get_reference(alias=alias, version=version)
        pipeline = self.pipeline_factory()
        artifact_path = self._artifact_downloader(reference.source)
        pipeline.load_artifact(artifact_path)

        expected_schema = build_feature_schema(features_config)
        if expected_schema.schema_version != pipeline.feature_schema.schema_version:
            from tracking.promotion import ModelValidationError

            raise ModelValidationError(
                "Loaded model feature schema version does not match runtime schema: "
                f"runtime={expected_schema.schema_version}, loaded={pipeline.feature_schema.schema_version}"
            )

        return {"reference": reference, "pipeline": pipeline}

    def resolve_model_source(self, run_id: str) -> str:
        artifacts = self.client.list_artifacts(run_id, "artifacts")
        joblib_artifacts = [artifact.path for artifact in artifacts if artifact.path.endswith(".joblib")]
        if not joblib_artifacts:
            raise ModelRegistryError("No joblib model artifact was logged for the supplied run")
        return f"runs:/{run_id}/{joblib_artifacts[0]}"

    def safe_get_reference(self, alias: str) -> Optional[ModelReference]:
        try:
            return self.get_reference(alias=alias)
        except ModelRegistryError:
            return None

    def get_reference(self, alias: Optional[str] = None, version: Optional[str] = None) -> ModelReference:
        self.require_available()
        if alias:
            try:
                model_version = self.client.get_model_version_by_alias(self.model_name, alias)
                return self.to_reference(model_version, alias=alias)
            except Exception as alias_exc:
                model_version = self.find_version_by_run_id(alias)
                if model_version is None:
                    raise ModelRegistryError(
                        f"'{alias}' is neither an assigned alias for model '{self.model_name}' "
                        "nor a run ID with a registered model version"
                    ) from alias_exc
                return self.to_reference(model_version, alias=None)
        if version:
            model_version = self.client.get_model_version(self.model_name, str(version))
            return self.to_reference(model_version, alias=None)
        raise ModelRegistryError("A model alias or explicit model version is required")

    def find_version_by_run_id(self, run_id: str) -> Optional[Any]:
        """Resolve a model version from either an MLflow tracking run ID or the
        training service's own `local_run_id` tag (see TrainingService._build_run_tags).
        Model IDs surfaced to CLI/backend callers are the local run ID, not the
        MLflow-native one, so a straight `run_id=` filter alone would miss them.
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}' and run_id='{run_id}'")
        except Exception:
            versions = []
        if not versions:
            try:
                runs = self.client.search_runs(
                    experiment_ids=[self.experiment_id()],
                    filter_string=f"tags.local_run_id = '{run_id}'",
                    max_results=1,
                )
                if runs:
                    versions = self.client.search_model_versions(
                        f"name='{self.model_name}' and run_id='{runs[0].info.run_id}'"
                    )
            except Exception:
                versions = []
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version))

    def experiment_id(self) -> str:
        experiment = self.client.get_experiment_by_name(settings.mlflow_experiment_name)
        if experiment is None:
            raise ModelRegistryError(f"MLflow experiment '{settings.mlflow_experiment_name}' not found")
        return experiment.experiment_id

    def to_reference(self, model_version: Any, alias: Optional[str]) -> ModelReference:
        tags = dict(getattr(model_version, "tags", {}) or {})
        return ModelReference(
            name=model_version.name,
            version=str(model_version.version),
            alias=alias,
            run_id=model_version.run_id,
            source=model_version.source,
            tags={str(key): str(value) for key, value in tags.items()},
        )

    def load_json_artifact(self, run_id: str, artifact_path: str) -> Any:
        import json
        from pathlib import Path

        artifact = self.client.download_artifacts(run_id, artifact_path)
        return json.loads(Path(artifact).read_text(encoding="utf-8"))

    def ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.model_name)
        except Exception:
            self.client.create_registered_model(self.model_name)

    def _default_download_artifact(self, artifact_uri: str) -> str:
        if mlflow is None:  # pragma: no cover - guarded by require_available
            raise ModelRegistryError("mlflow is not installed")
        return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def coerce_metric(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
