from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
from imblearn.pipeline import Pipeline as ImbPipeline

from features.schema import FeatureSchema


class ArtifactStore:
    """Filesystem access for trained-model `.joblib` artifacts under `models_path`.

    Pure I/O: path resolution, save, and raw payload load. Feature-schema
    compatibility validation and in-memory state (best_model, label_encoder, ...)
    stay in MLPipeline -- this only knows how to read and write files.
    """

    def __init__(self, models_path: str):
        self.models_path = models_path
        os.makedirs(self.models_path, exist_ok=True)

    def artifact_path(self, run_id: str, algorithm: str) -> str:
        return os.path.join(self.models_path, f"{algorithm}_{run_id}.joblib")

    def save(
        self,
        artifact_path: str,
        algorithm: str,
        pipeline: ImbPipeline,
        feature_schema: FeatureSchema,
        metadata: Dict[str, Any],
        label_encoder_classes: List[str],
    ) -> str:
        payload = {
            "algorithm": algorithm,
            "pipeline": pipeline,
            "feature_schema": feature_schema.to_dict(),
            "metadata": metadata,
            "label_encoder_classes": label_encoder_classes,
        }
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, artifact_path)
        return artifact_path

    def resolve_path(self, artifact_reference: str) -> str:
        if artifact_reference in (None, "", "latest"):
            return self.latest_path()
        candidate = Path(artifact_reference)
        if candidate.exists():
            return str(candidate)

        model_dir = Path(self.models_path)
        matches = sorted(model_dir.glob(f"*{artifact_reference}*.joblib"))
        if matches:
            return str(matches[-1])
        raise FileNotFoundError(f"Artifact not found: {artifact_reference}")

    def latest_path(self) -> str:
        matches = sorted(Path(self.models_path).glob("*.joblib"))
        if not matches:
            raise FileNotFoundError("No packaged model artifacts were found")
        return str(matches[-1])

    def load_payload(self, artifact_reference: str) -> Dict[str, Any]:
        return joblib.load(self.resolve_path(artifact_reference))
