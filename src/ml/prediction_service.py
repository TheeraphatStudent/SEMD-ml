from __future__ import annotations

from typing import Any, Dict, Optional

from core import get_logger
from monitoring.store import monitoring_store
from tracking.model_registry import CachedChampionModelLoader

logger = get_logger(__name__)


class PredictionService:
    def __init__(self) -> None:
        self.current_model_id: Optional[str] = None
        self.model_loader = CachedChampionModelLoader()

    def load_model(self, artifact_reference: str) -> bool:
        self.model_loader.load(selector=artifact_reference)
        self.current_model_id = artifact_reference
        return True

    def execute_prediction(self, job_data: Dict[str, Any], input_source: str = "queue") -> Dict[str, Any]:
        url = job_data.get("url")
        artifact_reference = job_data.get("model_id")
        if not url:
            raise ValueError("No URL provided for prediction")
        if artifact_reference and artifact_reference != self.current_model_id:
            self.load_model(artifact_reference)
        elif self.current_model_id is None:
            self.load_model("champion")
        result = self.model_loader.predict(url, selector=artifact_reference or self.current_model_id)
        result["prediction_id"] = self._record_event(result, input_source=input_source)
        return result

    def batch_predict(self, job_data: Dict[str, Any], input_source: str = "queue") -> Dict[str, Any]:
        urls = job_data.get("urls") or []
        artifact_reference = job_data.get("model_id")
        if artifact_reference and artifact_reference != self.current_model_id:
            self.load_model(artifact_reference)
        elif self.current_model_id is None:
            self.load_model("champion")
        return {"predictions": [self.execute_prediction({"url": url}, input_source=input_source) for url in urls]}

    def _record_event(self, result: Dict[str, Any], input_source: str) -> Optional[str]:
        # Best-effort: a monitoring-store failure must never break a prediction response, but it
        # must not be silent either (see docs/final-handoff.md's note on unlogged swallowed exceptions).
        try:
            return monitoring_store.record_event(
                url=result["url"],
                prediction=result["prediction"],
                confidence=result.get("confidence"),
                model_version=result.get("model_version"),
                model_alias=result.get("model_alias"),
                feature_schema_version=result.get("feature_schema_version"),
                prediction_latency_ms=result.get("prediction_time_ms"),
                input_source=input_source,
            )
        except Exception as exc:
            logger.warning("Failed to record prediction monitoring event: %s", exc)
            return None


prediction_service = PredictionService()
