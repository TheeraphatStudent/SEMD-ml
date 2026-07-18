from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_probabilities(pipeline: ImbPipeline, X_eval: pd.DataFrame) -> Optional[np.ndarray]:
    estimator = pipeline.named_steps["estimator"]
    if hasattr(estimator, "predict_proba"):
        return pipeline.predict_proba(X_eval)
    return None


def roc_auc(y_eval: pd.Series, probabilities: Optional[np.ndarray]) -> Optional[float]:
    if probabilities is None or len(np.unique(y_eval)) < 2:
        return None
    return float(roc_auc_score(y_eval, probabilities[:, 1]))


def measure_prediction_latency_ms(pipeline: ImbPipeline, X_eval: pd.DataFrame) -> float:
    sample_count = min(10, len(X_eval))
    if sample_count == 0:
        return 0.0
    durations = []
    for idx in range(sample_count):
        start = time.perf_counter()
        pipeline.predict(X_eval.iloc[[idx]])
        durations.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(durations))


def evaluate_model(pipeline: ImbPipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, Any]:
    predictions = pipeline.predict(X_eval)
    probabilities = predict_probabilities(pipeline, X_eval)
    cm = confusion_matrix(y_eval, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    latency_ms = measure_prediction_latency_ms(pipeline, X_eval)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "malicious_precision": float(precision_score(y_eval, predictions, pos_label=1, zero_division=0)),
        "malicious_recall": float(recall_score(y_eval, predictions, pos_label=1, zero_division=0)),
        "malicious_f1": float(f1_score(y_eval, predictions, pos_label=1, zero_division=0)),
        "macro_precision": float(precision_score(y_eval, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_eval, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_eval, predictions, average="macro", zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": cm.tolist(),
        "prediction_latency_ms": latency_ms,
    }
    metrics["roc_auc"] = roc_auc(y_eval, probabilities)
    return metrics
