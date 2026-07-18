from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder

from features import feature_extractor
from features.schema import FeatureSchema
from ml.evaluation import predict_probabilities


def predict(
    pipeline: ImbPipeline, feature_schema: FeatureSchema, label_encoder: LabelEncoder, url: str
) -> Dict[str, Any]:
    if pipeline is None or feature_schema is None:
        raise ValueError("No model loaded. Train or load a model first.")

    start = time.perf_counter()
    features = feature_extractor.extract(url)
    features = feature_schema.align_record(features)
    X = pd.DataFrame([features], columns=feature_schema.feature_names)
    probabilities = predict_probabilities(pipeline, X)
    prediction = pipeline.predict(X)[0]
    confidence = 1.0
    if probabilities is not None:
        confidence = float(np.max(probabilities[0]))
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "url": url,
        "prediction": prediction_label,
        "is_malicious": prediction_label == "malicious",
        "confidence": confidence,
        "feature_schema_version": feature_schema.schema_version,
        "prediction_time_ms": round(elapsed_ms, 2),
    }
