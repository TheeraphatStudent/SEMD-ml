# MLflow Tracking

## Scope

Section 5 adds MLflow Tracking to the SEMD ML training and evaluation pipeline. It does not implement automatic production promotion.

## Environment variables

Use these variables instead of hardcoded deployment values:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=semd-url-classification
MLFLOW_REGISTERED_MODEL_NAME=semd-malicious-url-detector
MLFLOW_ARTIFACT_ROOT=./artifacts/mlflow
```

For containerized training, Compose overrides `MLFLOW_TRACKING_URI` with `MLFLOW_CONTAINER_TRACKING_URI=http://mlflow:5000` so the ML container reaches the tracking server over the service network.

## Tracking behavior

- `tracking/mlflow_tracker.py` configures the tracking URI and selects or creates the configured experiment.
- Run names come from the CLI `--run-name` flag or default to `training_YYYYMMDD_HHMMSS` / `evaluation_YYYYMMDD_HHMMSS`.
- Run tags capture run kind, requested algorithms, dataset version/hash, git SHA, registered model name, and the explicit autologging decision.
- Configuration failures are handled gracefully: training continues and the tracker records the last MLflow error in the returned payload.

## Autologging decision

MLflow autologging is intentionally disabled.

Reason:

- The project requires dataset versioning, dataset hash, balancing method, feature schema version, dataset quality outputs, and custom artifact packaging.
- `mlflow.sklearn.autolog()` would still miss required project metadata and would make the resulting run shape harder to control across multi-algorithm training runs.

The implementation logs all required metadata explicitly and does not rely on autologging.

## Logged parameters

- algorithm
- requested_algorithms
- hyperparameters
- random_state
- dataset_version
- dataset_hash
- sample_size
- train_size
- validation_size
- test_size
- balancing_method
- scaling_method
- feature_schema_version
- feature_count
- class_distribution
- git_commit_sha
- python_version
- dataset_files
- dataset_sources
- registered_model_name

## Logged metrics

Primary run metrics are recorded for the selected best model:

- train_accuracy
- validation_accuracy
- test_accuracy
- malicious_precision
- malicious_recall
- malicious_f1
- macro_precision
- macro_recall
- macro_f1
- roc_auc
- false_positive_rate
- false_negative_rate
- training_duration_seconds
- prediction_latency_ms
- cross_validation_mean
- cross_validation_std

Per-algorithm metrics are also logged with an `<algorithm>_...` prefix.

## Logged artifacts

- `classification_report.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `feature_schema.json`
- `training_configuration.json`
- `dataset_quality_report.json`
- `dataset_metadata.json`
- model `.joblib` artifact
- `requirements.txt`
- `sample_predictions.json`

## Compose integration

- `docker/docker-compose.yml` starts `mlflow` plus the `ml-service`.
- MLflow persists backend state in `mlflow_data/`.
- Artifacts persist in `artifacts/`.
- The ML service waits for the MLflow health check before startup.
- The UI is exposed on `http://localhost:5000`.

## Verification

Recommended checks:

```bash
cd semd-ml
cp .env.example .env
docker compose -f docker/docker-compose.yml up -d mlflow
curl http://localhost:5000
cd src
uv run python main.py train --algorithms random_forest --run-name mlflow-smoke
uv run pytest tests/integration/test_mlflow_tracking.py
```
