from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from core import settings
from tracking.promotion import GateResult, ModelValidationError, Promotion
from tracking.registry import ModelReference, ModelRegistryError, Registry

if TYPE_CHECKING:
    from ml.ml_pipeline import MLPipeline

__all__ = [
    "ModelRegistryError",
    "ModelValidationError",
    "GateResult",
    "ModelReference",
    "ModelRegistryManager",
    "CachedChampionModelLoader",
]


class ModelRegistryManager:
    """Facade preserving the combined registry+promotion API every existing
    caller (CLI commands, CachedChampionModelLoader, tests) already depends on.

    Internally composes tracking.registry.Registry (pure CRUD) and
    tracking.promotion.Promotion (gate/validation policy) -- see
    docs/refactoring-plan.md T08. Splitting them lets tracking run-logging
    (tracking/mlflow_tracker.py) stay independent of promotion policy: it no
    longer imports anything from this module or from tracking/promotion.py.
    """

    def __init__(
        self,
        client: Any | None = None,
        pipeline_factory: Optional[Callable[[], "MLPipeline"]] = None,
        artifact_downloader: Optional[Callable[[str], str]] = None,
    ) -> None:
        # pipeline_factory defaults to MLPipeline, deferred-imported inside
        # Registry.__init__ rather than at this module's top level -- see the
        # same pattern (and why) in tracking/registry.py.
        self._registry = Registry(
            client=client, pipeline_factory=pipeline_factory, artifact_downloader=artifact_downloader
        )
        self._promotion = Promotion(self._registry)

    @property
    def client(self) -> Any:
        return self._registry.client

    @client.setter
    def client(self, value: Any) -> None:
        self._registry.client = value

    @property
    def model_name(self) -> str:
        return self._registry.model_name

    @property
    def candidate_alias(self) -> str:
        return self._registry.candidate_alias

    @property
    def champion_alias(self) -> str:
        return self._registry.champion_alias

    @property
    def previous_champion_alias(self) -> str:
        return self._registry.previous_champion_alias

    @property
    def pipeline_factory(self) -> Callable[[], MLPipeline]:
        return self._registry.pipeline_factory

    @property
    def last_error(self) -> Optional[str]:
        return self._registry.last_error

    @property
    def available(self) -> bool:
        return self._registry.available

    def register_candidate(self, run_id: str) -> Dict[str, Any]:
        return self._registry.register_candidate(run_id)

    def rollback_to_previous_champion(self) -> Dict[str, Any]:
        return self._registry.rollback_to_previous_champion()

    def load_reference(self, alias: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Any]:
        return self._registry.load_reference(alias=alias, version=version)

    def validate_candidate(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        return self._promotion.validate_candidate(model_version)

    def promote_candidate(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        return self._promotion.promote_candidate(model_version)


class CachedChampionModelLoader:
    def __init__(
        self,
        registry_manager: Optional[Any] = None,
        pipeline_factory: Optional[Callable[[], "MLPipeline"]] = None,
    ) -> None:
        if pipeline_factory is None:
            from ml.ml_pipeline import MLPipeline as _MLPipeline

            pipeline_factory = _MLPipeline
        self.registry_manager = registry_manager or ModelRegistryManager(pipeline_factory=pipeline_factory)
        self.pipeline_factory = pipeline_factory
        self._cached_pipeline: Optional["MLPipeline"] = None
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

        is_cached_champion = self._cached_pipeline is not None and self._cached_reference is not None
        if alias == settings.mlflow_alias_champion and is_cached_champion:
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
