from __future__ import annotations

from typing import Any, Dict, Optional

from tracking.model_registry import CachedChampionModelLoader


class PredictionService:
    def __init__(self) -> None:
        self.current_model_id: Optional[str] = None
        self.model_loader = CachedChampionModelLoader()

    def load_model(self, artifact_reference: str) -> bool:
        self.model_loader.load(selector=artifact_reference)
        self.current_model_id = artifact_reference
        return True

    def execute_prediction(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        url = job_data.get("url")
        artifact_reference = job_data.get("model_id")
        if not url:
            raise ValueError("No URL provided for prediction")
        if artifact_reference and artifact_reference != self.current_model_id:
            self.load_model(artifact_reference)
        elif self.current_model_id is None:
            self.load_model("champion")
        return self.model_loader.predict(url, selector=artifact_reference or self.current_model_id)

    def batch_predict(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        urls = job_data.get("urls") or []
        artifact_reference = job_data.get("model_id")
        if artifact_reference and artifact_reference != self.current_model_id:
            self.load_model(artifact_reference)
        elif self.current_model_id is None:
            self.load_model("champion")
        return {"predictions": [self.execute_prediction({"url": url}) for url in urls]}


prediction_service = PredictionService()
