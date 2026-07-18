from __future__ import annotations

from typing import Any, Dict, Optional

from core import get_logger
from monitoring.store import monitoring_store
from pipelines.prediction_pipeline import PredictionPipeline

logger = get_logger(__name__)


class PredictionService:
    def __init__(self) -> None:
        self._pipeline = PredictionPipeline()

    @property
    def current_model_id(self) -> Optional[str]:
        return self._pipeline.current_model_id

    @current_model_id.setter
    def current_model_id(self, value: Optional[str]) -> None:
        self._pipeline.current_model_id = value

    @property
    def model_loader(self) -> Any:
        return self._pipeline.model_loader

    @model_loader.setter
    def model_loader(self, value: Any) -> None:
        self._pipeline.model_loader = value

    def load_model(self, artifact_reference: str) -> bool:
        return self._pipeline.load_model(artifact_reference)

    def execute_prediction(self, job_data: Dict[str, Any], input_source: str = "queue") -> Dict[str, Any]:
        result = self._pipeline.predict(job_data.get("url"), model_id=job_data.get("model_id"))
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
