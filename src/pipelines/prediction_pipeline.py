from __future__ import annotations

from typing import Any, Dict, Optional


class PredictionPipeline:
    """Model resolution + inference for a single URL: reuse the already-loaded
    model unless a different one is requested, otherwise fall back to champion.

    No monitoring-event recording or batch orchestration here -- those are
    PredictionService's job (see ml/prediction_service.py).
    """

    def __init__(self, model_loader: Optional[Any] = None) -> None:
        if model_loader is None:
            # Deferred import: tracking.model_registry -> tracking/__init__ ->
            # mlflow_tracker -> ml.ml_pipeline -> (ml package init) ->
            # ml.prediction_service -> this module is an existing circular
            # dependency in this codebase (see tests/unit/test_queue_worker.py's
            # note on tracking/mlflow_tracker.py <-> tracking/model_registry.py).
            # A module-level import here reliably breaks whenever something
            # imports pipelines.prediction_pipeline before the ml package has
            # finished initializing. Importing lazily, only once a
            # PredictionPipeline is actually constructed, avoids adding a new
            # trigger for that pre-existing cycle.
            from tracking.model_registry import CachedChampionModelLoader

            model_loader = CachedChampionModelLoader()
        self.model_loader = model_loader
        self.current_model_id: Optional[str] = None

    def load_model(self, artifact_reference: str) -> bool:
        self.model_loader.load(selector=artifact_reference)
        self.current_model_id = artifact_reference
        return True

    def predict(self, url: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        if not url:
            raise ValueError("No URL provided for prediction")
        if model_id and model_id != self.current_model_id:
            self.load_model(model_id)
        elif self.current_model_id is None:
            self.load_model("champion")
        return self.model_loader.predict(url, selector=model_id or self.current_model_id)
