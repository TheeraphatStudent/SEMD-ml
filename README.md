# SEMD ML Service - Malicious URL Detection

Advanced ML microservice with continuous fine-tuning, dynamic feature engineering, and multi-stage feature selection pipeline.

## Project Structure

The project follows a modular architecture. All CLI commands run from `src/`, but there is also a `makefile` at the repo root that wraps them for convenience (see [CLI Usage](#cli-usage)).

```
semd-ml/
├── makefile                        # Infra targets (venv, start/stop/status) + ML CLI targets (train, predict, ...)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── CLAUDE.md                        # Module map and data flow reference for Claude Code
├── .env.example                     # Environment configuration template
│
├── docker/
│   ├── docker-compose.yml           # ML service + MLflow orchestration
│   └── Dockerfile                   # ML service container
│
├── dataset/                         # Raw/extracted training datasets (see src/dataset/ below)
├── models/                          # Trained model artifacts (model_*.pkl, scaler_*.pkl, etc.)
├── reports/                         # Training/evaluation/prediction reports (generated, gitignored)
├── mlflow_data/                     # MLflow backend store (generated)
│
└── src/                             # Source code — working directory for all CLI commands
    ├── __init__.py                  # Backward compatibility exports
    ├── main.py                      # CLI entry point (delegates to cli.main)
    ├── verify_imports.py            # Smoke-tests that every module imports cleanly
    │
    ├── core/                        # Core configuration and utilities
    │   ├── config.py                # Settings (.env) and feature configuration
    │   ├── logger.py                # Logging setup
    │   ├── archive_utils.py         # Archive extraction helpers for data-migrate
    │   └── reporting.py             # Shared JSON result output helper
    │
    ├── features/                    # Feature engineering
    │   ├── features.yaml            # Feature definitions
    │   └── feature_extractor.py     # Feature extraction logic
    │
    ├── data/                        # Data loading and preprocessing
    │   ├── data_dict.yaml           # Column/label mapping config
    │   └── dataset_pipeline.py      # Dataset loading, validation, balancing
    │
    ├── ml/                          # Machine learning pipeline
    │   ├── ml_pipeline.py           # ML training and evaluation
    │   ├── training_service.py      # Training orchestration
    │   └── prediction_service.py    # Prediction service
    │
    ├── infra/                       # Infrastructure clients
    │   ├── database.py              # PostgreSQL client
    │   └── redis_client.py          # Redis client
    │
    ├── tracking/                    # Experiment tracking
    │   └── mlflow_tracker.py        # MLflow integration
    │
    ├── queues/                      # Redis queue management
    │   └── queue_manager.py         # Queue push/pop, status reporting
    │
    ├── workers/                     # Long-running queue consumers
    │   └── queue_worker.py          # Redis queue worker
    │
    ├── cli/                         # Command-line interface
    │   ├── main.py                  # argparse setup + command dispatch
    │   ├── common.py                # Shared helpers (emit_result, validate_algorithms)
    │   └── commands/                # One module per command group
    │       ├── train.py             # cmd_train, cmd_train_obo
    │       ├── predict.py           # cmd_predict, cmd_predict_test
    │       ├── evaluate.py          # cmd_evaluate
    │       ├── feature_engineering.py
    │       ├── worker.py            # cmd_worker, cmd_queue_status
    │       └── migrate.py           # cmd_data_migrate, cmd_data_migrate_feature
    │
    └── dataset/
        ├── store/                   # Raw dataset archives (zip/gz)
        ├── raw/                     # Extracted CSVs used for training (data-migrate output)
        ├── feature/                 # Feature reference CSVs (brand keywords, suspicious TLDs, etc.)
        ├── test/                    # Sample CSVs for predict-test
        └── script/                  # Dataset download scripts (HuggingFace, Cloudflare)
```

See `CLAUDE.md` for the full data flow (training/prediction) and configuration file details.

## Documentation

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Target module structure and migration plan |
| [`docs/operations.md`](docs/operations.md) | Diagrams: system architecture, dataset lifecycle, training, MLflow lifecycle, promotion, inference, container startup |
| [`docs/dataset-pipeline.md`](docs/dataset-pipeline.md) | Dataset loading, validation, cleaning, balancing |
| [`docs/feature-schema.md`](docs/feature-schema.md) | Feature list, versioning, schema alignment |
| [`docs/mlflow-tracking.md`](docs/mlflow-tracking.md) | What gets logged per training run |
| [`docs/model-registry.md`](docs/model-registry.md) | Registration, promotion gates, aliases |
| [`docs/model-serving.md`](docs/model-serving.md) | Champion loading, caching, local fallback |
| [`docs/model-evaluation.md`](docs/model-evaluation.md) | Metrics computed during training/evaluation |
| [`docs/rollback.md`](docs/rollback.md) | Reverting a promoted model |
| [`docs/retraining.md`](docs/retraining.md) | The manual, human-approved retraining workflow |
| [`docs/testing.md`](docs/testing.md) | Test suite layout, coverage map, quality commands |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common setup/test/MLflow/registry failures |

## Setup

### Python Environment

```bash
make venv
```

This creates `.venv` with `uv` and installs the project (editable) from `pyproject.toml`,
including the `tracking` extra (mlflow — required by `predict`/`register`/`promote`) and
`xgboost`. Manual equivalent:

```bash
uv sync --extra tracking --extra xgboost --group dev
```

`requirements.txt` still exists but is no longer the dev install source — it's only what
the Docker image installs from until that's migrated too (see
[`docs/refactoring-plan.md`](docs/refactoring-plan.md) T13).

### Configuration

```bash
cp .env.example .env
```

## CLI Usage

All CLI commands are implemented in `src/main.py` (via `cli/`) and are designed to run **from the `src/` directory**:

```bash
cd src
uv run python main.py <command> ...
```

The root `makefile` wraps every subcommand so you don't have to `cd` manually — run these from the repo root instead:

| Command | `make` target | Direct equivalent (from `src/`) |
|---|---|---|
| Train models | `make train DATASET_FILES=dataset/raw ALGORITHMS="decision_tree random_forest xgboost svm" RUN_NAME=my_run` | `uv run python main.py train --dataset-files dataset/raw --algorithms decision_tree random_forest xgboost svm --run-name my_run` |
| Train one-by-one per dataset in store | `make train-obo ALGORITHMS=random_forest RUN_NAME=obo_run` | `uv run python main.py train-obo --algorithms random_forest --run-name obo_run` |
| Predict a URL | `make predict URL="https://example.com" MODEL_ID=<run_id>` | `uv run python main.py predict --url "https://example.com" --model-id <run_id>` |
| Batch-test URLs with timing/metrics | `make predict-test CSV=urls.csv MODEL_ID=<run_id>` | `uv run python main.py predict-test --csv urls.csv --model-id <run_id>` |
| Evaluate models | `make evaluate DATASET_FILES=dataset/raw ALGORITHMS="random_forest xgboost"` | `uv run python main.py evaluate --dataset-files dataset/raw --algorithms random_forest xgboost` |
| Feature engineering analysis | `make feature-engineering URL="https://example.com"` | `uv run python main.py feature-engineering --url "https://example.com"` |
| Start queue worker | `make worker MODE=combined` | `uv run python main.py worker --mode combined` |
| Check Redis queue status | `make queue-status` | `uv run python main.py queue-status` |
| Extract raw datasets from archives | `make data-migrate` | `uv run python main.py data-migrate` |
| Migrate feature reference CSVs | `make data-migrate-feature` | `uv run python main.py data-migrate-feature` |
| Verify all imports | `make verify-imports` | `uv run python verify_imports.py` |
| Any other/less common flags | `make cli ARGS='predict --url ... --compare'` | — |

Every `make` target accepts extra raw flags via `ARGS='...'`, e.g. `make train ALGORITHMS=svm ARGS='--balance smote'`.

### Notes

- `--output`/`OUTPUT` writes results as JSON to `../reports/<file>` (relative to `src/`) instead of only printing to stdout.
- `predict` supports `--url`, `--urls` (space-separated), or `--url-file` (one URL per line); `--compare` prints a feature comparison table for 2-5 URLs.
- `predict-test` accepts `--url`, `--urls`, or `--csv` (looked up under `dataset/test/` if a relative path), and reports success rate, per-URL timing, and throughput.

## Redis worker

### Start Queue Worker

```bash
# Combined mode (training + prediction)
make worker MODE=combined

# Training only
make worker MODE=training

# Prediction only
make worker MODE=prediction
```

### Queue train

To submit training jobs asynchronously, push JSON messages to the Redis queue `ml_training_queue`. Example:

```bash
redis-cli LPUSH ml_training_queue '{
  "service_conf_id": 1,
  "dataset_files": ["dataset/malicious_url_train.csv"],
  "algorithms": ["random_forest", "xgboost"],
  "run_name": "async_training_run",
  "balance_method": "smote"
}'
```

For predictions, push to `ml_prediction_queue`:

```bash
redis-cli LPUSH ml_prediction_queue '{
  "url": "https://example.com",
  "user_id": "user123",
  "model_id": "run_abc123"
}'
```

### Check Queue Status

```bash
make queue-status
```

## Services Setup

```bash
# Start all Docker services
make start

# Check status
make status

# View logs
make logs LOG_SERVICE=mlflow
make logs LOG_SERVICE=backend

# Stop services
make stop

# Restart services
make restart
```

## ML Flow

### Fix permissions

```bash
make mlflow-permissions
```

## Configuration Options

### Feature Selection

```env
ENABLE_FEATURE_SELECTION=true
FEATURE_SELECTION_K=50
ENABLE_CORRELATION_FILTER=true
CORRELATION_THRESHOLD=0.95
ENABLE_VARIANCE_THRESHOLD=true
VARIANCE_THRESHOLD=0.01
ENABLE_MUTUAL_INFORMATION=true
MUTUAL_INFO_THRESHOLD=0.01
```

### Feature Importance

```env
ENABLE_FEATURE_IMPORTANCE=true
ENABLE_CLASS_WEIGHTING=true
CLASS_WEIGHT_MODE=soft
```

## Docker Deployment

```bash
make start
```

This starts:
- ML Service worker
- Redis
- PostgreSQL
- MLflow server

## Testing & Quality

```bash
uv sync --extra tracking --extra xgboost --group dev   # installs ruff, mypy, pytest + mlflow/xgboost extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run --extra tracking --extra xgboost pytest -v
```

See [`docs/testing.md`](docs/testing.md) for the full coverage map and known baseline limitations (some
pre-existing files are not yet `ruff format`-clean; `mypy` has a documented pre-existing error baseline).

## Dataset Resources

- https://urlhaus.abuse.ch/browse/
- https://www.phishtank.com/developer_info.php
- https://huggingface.co/datasets/JorgeGMM/malicious_urls
- https://huggingface.co/datasets/EustassKidman/malicious-url/viewer/default/train
