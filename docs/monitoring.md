# Prediction Monitoring

Prediction telemetry is stored **outside MLflow runs** — predictions never create an MLflow run
(`docs/final-handoff.md` flagged "no one-run-per-prediction misuse" as a thing to preserve). Instead, every
prediction is recorded as a row in a self-contained local SQLite store owned by `semd-ml` itself, independent of
the shared Postgres/Redis infrastructure (see `docs/final-handoff.md`'s note on the cross-module Redis-auth gap —
this store deliberately does not depend on that).

## Storage

- Module: `src/monitoring/store.py` (`MonitoringStore`, default singleton `monitoring_store`)
- Location: `MONITORING_DB_PATH` (default `<repo_root>/monitoring/monitoring.db`)
- Table `prediction_events`, one row per prediction:

| Column | Notes |
|---|---|
| `prediction_id` | primary key, returned by `predict` / `predict-test` |
| `url` | raw URL — kept alongside the hash because retraining needs real `(url, label)` pairs to re-featurize; a stored feature vector would silently drift from a later `feature_schema_version` |
| `url_hash` | SHA-256 of the lowercased, stripped URL — stable dedup/lookup key |
| `prediction`, `confidence` | model output |
| `model_version`, `model_alias` | which registry version/alias served this prediction |
| `feature_schema_version` | schema version active at prediction time |
| `prediction_latency_ms` | per-prediction latency |
| `input_source` | `cli` \| `queue` (set by the caller — `cli/commands/predict.py` and `workers/queue_worker.py`) |
| `created_at` | ISO-8601 UTC timestamp |
| `user_feedback` | `reported_incorrect` \| `confirmed_correct` \| null — a user-facing flag, not a label |
| `admin_reviewed_label` | `benign` \| `malicious` \| null — the operator-approved ground truth label |
| `admin_reviewed_at` | set when `admin_reviewed_label` is set |

Recording is best-effort: a monitoring-store failure never breaks a prediction response, but it is **not**
silent — failures are logged via `logger.warning` in `PredictionService._record_event`.

## Attaching feedback

```bash
uv run semd-ml feedback --prediction-id <id> --status reported_incorrect
uv run semd-ml review --prediction-id <id> --label malicious
```

`feedback` records what an end user said about a prediction. `review` records what an admin/analyst determined
after investigation — this is the "approved feedback" that retraining consumes (see `docs/retraining.md`).

## Metrics

```bash
uv run semd-ml monitor [--since <iso-timestamp>] [-o report.json]
```

`src/monitoring/metrics.py::compute_monitoring_metrics` aggregates:

- `prediction_count`, `malicious_ratio`, `mean_confidence`
- `latency_percentiles_ms` (`p50`/`p90`/`p99`)
- `user_report_count` — count of `user_feedback == "reported_incorrect"`
- `admin_correction_count` — count of reviewed events where `admin_reviewed_label != prediction`
- `reviewed_count` — how many events have been admin-reviewed at all
- `estimated_false_positive_rate` — of reviewed events predicted `malicious`, the fraction admin-labeled `benign`
- `estimated_false_negative_rate` — of reviewed events predicted `benign`, the fraction admin-labeled `malicious`

**These FPR/FNR figures are estimates over the reviewed subset only** (`reviewed_count` out of `prediction_count`),
not population rates — admins tend to review reported or suspicious predictions, not a random sample, so the
reviewed subset is biased. The report always includes `reviewed_count` and an `estimate_note` alongside the
numbers so this isn't lost in downstream consumption.

## What this does not do

- No dashboard/scheduler polls this automatically — `monitor` is a CLI command a person runs.
- No prediction ever creates an MLflow run; this store is the only place prediction telemetry lives.

## Known gap: not yet container-persistent

`monitoring_db_path` defaults to `<repo_root>/monitoring/monitoring.db`, which resolves to `/app/monitoring/...`
inside the `ml-service` container. `docker/docker-compose.yml` does not currently bind-mount `monitoring/` the way
it mounts `models/`, `reports/`, `src/dataset/`, and `artifacts/` — so telemetry recorded by a containerized
worker is lost when the container is removed. This mirrors the artifact-root mount gap noted in
`docs/final-handoff.md` §9. Add `- ../monitoring:/app/monitoring` to the `ml-service` volumes list to close it;
not done here since it wasn't exercised against a live container in this pass.
