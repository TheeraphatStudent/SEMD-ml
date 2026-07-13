from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from core import get_logger, settings
from monitoring.store import MonitoringStore, monitoring_store

logger = get_logger(__name__)


def build_feedback_dataset(
    store: Optional[MonitoringStore] = None,
    since: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn admin-approved prediction feedback into a labeled CSV for retraining.

    Only events with a non-null `admin_reviewed_label` are used — that field is the operator-approved
    ground truth ("approved feedback"), distinct from the model's original `prediction` and from raw
    `user_feedback` (a user-reported flag, not a label). Re-featurizing from the stored `url` at train
    time (rather than trusting any cached feature vector) keeps this immune to `feature_schema_version`
    drift between prediction time and retrain time.
    """
    store = store or monitoring_store
    events = store.get_events(since=since, reviewed_only=True)
    output_dir = output_dir or settings.monitoring_dataset_dir
    os.makedirs(output_dir, exist_ok=True)

    if not events:
        return {"path": None, "record_count": 0, "since": since}

    frame = pd.DataFrame(
        {
            "url": [event["url"] for event in events],
            "label": [event["admin_reviewed_label"] for event in events],
        }
    ).drop_duplicates(subset=["url"], keep="last")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    path = os.path.join(output_dir, f"feedback_retrain_{timestamp}.csv")
    frame.to_csv(path, index=False)
    logger.info("Wrote %d approved-feedback rows to %s", len(frame), path)
    return {"path": path, "record_count": int(len(frame)), "since": since}
