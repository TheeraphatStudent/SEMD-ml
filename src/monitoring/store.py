from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from core import settings

VALID_USER_FEEDBACK = {"reported_incorrect", "confirmed_correct"}
VALID_ADMIN_LABELS = {"benign", "malicious"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_events (
    prediction_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    url_hash TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL,
    model_version TEXT,
    model_alias TEXT,
    feature_schema_version TEXT,
    prediction_latency_ms REAL,
    input_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    user_feedback TEXT,
    admin_reviewed_label TEXT,
    admin_reviewed_at TEXT
)
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_url(url: str) -> str:
    return hashlib.sha256(url.strip().lower().encode("utf-8")).hexdigest()


class MonitoringStore:
    """Self-contained SQLite store for prediction telemetry, independent of MLflow runs.

    Raw URLs are kept alongside the hash: `url_hash` is a stable dedup/lookup key, but retraining
    needs real (url, label) pairs to re-featurize, and re-featurizing from a stored feature vector
    would silently drift from a later `feature_schema_version` — so the source URL is authoritative.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.monitoring_db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_event(
        self,
        *,
        url: str,
        prediction: str,
        confidence: Optional[float],
        model_version: Optional[str],
        model_alias: Optional[str],
        feature_schema_version: Optional[str],
        prediction_latency_ms: Optional[float],
        input_source: str,
        prediction_id: Optional[str] = None,
    ) -> str:
        prediction_id = prediction_id or uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_events (
                    prediction_id, url, url_hash, prediction, confidence,
                    model_version, model_alias, feature_schema_version,
                    prediction_latency_ms, input_source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    url,
                    hash_url(url),
                    prediction,
                    confidence,
                    model_version,
                    model_alias,
                    feature_schema_version,
                    prediction_latency_ms,
                    input_source,
                    _utc_now(),
                ),
            )
        return prediction_id

    def set_user_feedback(self, prediction_id: str, feedback: str) -> bool:
        if feedback not in VALID_USER_FEEDBACK:
            raise ValueError(f"user_feedback must be one of {sorted(VALID_USER_FEEDBACK)}, got {feedback!r}")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE prediction_events SET user_feedback = ? WHERE prediction_id = ?",
                (feedback, prediction_id),
            )
            return cursor.rowcount > 0

    def set_admin_review(self, prediction_id: str, label: str) -> bool:
        if label not in VALID_ADMIN_LABELS:
            raise ValueError(f"admin_reviewed_label must be one of {sorted(VALID_ADMIN_LABELS)}, got {label!r}")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE prediction_events SET admin_reviewed_label = ?, admin_reviewed_at = ? WHERE prediction_id = ?",
                (label, _utc_now(), prediction_id),
            )
            return cursor.rowcount > 0

    def get_event(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prediction_events WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_events(
        self,
        since: Optional[str] = None,
        reviewed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM prediction_events WHERE 1=1"
        params: List[Any] = []
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        if reviewed_only:
            query += " AND admin_reviewed_label IS NOT NULL"
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]


monitoring_store = MonitoringStore()
