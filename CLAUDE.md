# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEMD ML Service is a malicious URL detection system using classical ML algorithms (Decision Tree, Random Forest, XGBoost, SVM/SGD). It features a CLI interface, Redis-based async job queues, MLflow experiment tracking, and a multi-stage feature engineering pipeline.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**All CLI commands must be run from the `src/` directory.**

## Common Commands

### Data Migration (run before training)
```bash
# Extract raw datasets from archives in dataset/store/
python3 main.py data-migrate

# Migrate feature reference data (brand keywords, suspicious TLDs, etc.)
python3 main.py data-migrate-feature
```

### Training
```bash
# Train one or more algorithms
python3 main.py train \
  --dataset-files dataset/raw \
  --algorithms decision_tree random_forest xgboost svm \
  --run-name my_run \
  --output ../reports/results.json
```

Available algorithms: `decision_tree`, `random_forest`, `xgboost`, `svm`
Balance methods: `none`, `smote`, `oversampling`, `undersampling`

### Prediction
```bash
python3 main.py predict --url "https://example.com" --model-id <run_id>
python3 main.py predict --url-file urls.txt --model-id <run_id>
```

### Other CLI Commands
```bash
python3 main.py evaluate --dataset-files dataset/raw --algorithms random_forest
python3 main.py feature-engineering --url "https://example.com"
python3 main.py worker --mode combined   # combined | training | prediction
python3 main.py queue-status
python3 verify_imports.py
```

### Docker Services
```bash
docker-compose up -d   # starts ml-service, MLflow (port 5000), Redis, PostgreSQL
```

MLflow UI: http://localhost:5000

### MLflow Permissions Fix
```bash
sudo chown -R 1001:1001 /home/semd/.mlflow
```

## Architecture

### Module Structure (all under `src/`)

| Module | Purpose |
|--------|---------|
| `core/config.py` | `MLServiceSettings` (pydantic-settings, reads `.env`) + `FeaturesConfig` (reads `features/features.yaml`). Global singletons: `settings`, `features_config`. |
| `features/feature_extractor.py` | `FeatureExtractor` — extracts ~80+ numeric features from a URL string. Feature lists loaded from `features/features.yaml`; feature reference data (brand keywords, suspicious TLDs, etc.) loaded from CSV files in `dataset/feature/raw/`. Global singleton: `feature_extractor`. |
| `data/dataset_pipeline.py` | `DatasetPipeline` — loads/merges CSV datasets, normalizes labels, runs parallel feature extraction via `multiprocessing.Pool`, detects class imbalance, applies balancing (SMOTE/over/under-sampling). Global singleton: `dataset_pipeline`. |
| `ml/ml_pipeline.py` | `MLPipeline` — multi-stage feature selection (correlation filter → variance filter → mutual info → SelectKBest), `RandomizedSearchCV` training, model evaluation, artifact save/load (joblib `.pkl`). Global singleton: `ml_pipeline`. |
| `ml/training_service.py` | Orchestrates full training flow: dataset prep → feature selection → train & compare → MLflow logging → save artifacts. |
| `ml/prediction_service.py` | Loads model artifacts by run ID, runs `feature_extractor.extract()`, applies stored feature selector, returns class + confidence. |
| `tracking/mlflow_tracker.py` | `MLflowTracker` wraps MLflow: starts/ends runs, logs params/metrics, registers sklearn models. Global singleton: `mlflow_tracker`. |
| `infra/` | `database.py` (PostgreSQL via psycopg2), `redis_client.py` (Redis). |
| `queues/queue_manager.py` | Redis queue push/pop for async training and prediction jobs. |
| `workers/queue_worker.py` | Long-running worker process consuming `ml_training_queue` and `ml_prediction_queue`. |
| `cli/cli_commands.py` | Implements all CLI subcommands (`cmd_train`, `cmd_predict`, etc.). |
| `dataset/script/` | Utility scripts for downloading datasets from HuggingFace and Cloudflare. |

### Key Data Flow

**Training:**
1. `DatasetPipeline.prepare_dataset()` → load CSVs → extract URL features (parallel) → balance classes
2. `MLPipeline.feature_selection()` → multi-stage filtering
3. `MLPipeline.train_and_compare_models()` → `RandomizedSearchCV` per algorithm → pick best F1
4. `MLflowTracker` logs all params/metrics/models
5. `MLPipeline.save_artifacts()` → saves `.pkl` files to `../models/`

**Prediction:**
1. Load model artifacts by `run_id` from `../models/`
2. `feature_extractor.extract(url)` → apply stored feature selector → `model.predict()`

### Configuration Files

- `src/features/features.yaml` — defines feature names, types, groups, and class emphasis weights. **Must be kept in sync with `FeatureExtractor` extraction methods.** Adding a feature requires both a YAML entry and extraction logic.
- `src/data/data_dict.yaml` — maps dataset column names to `url`/`label` fields; maps raw label values (0, 1, "phishing", etc.) to canonical classes (`benign`/`malicious`).
- `src/dataset/feature/dataset_feature.yaml` — config for migrating feature reference CSVs.
- `.env` — all runtime settings (Redis, PostgreSQL, MLflow URI, feature selection toggles).

### Paths Convention

The service is designed to run with `src/` as the working directory:
- Datasets: `./dataset/raw/` (relative to `src/`)
- Models: `../models/` (one level up from `src/`)
- Reports: `../reports/`
- MLflow data: `../mlflow_data/`

### Feature Reference Data

`FeatureExtractor` loads lookup sets from `dataset/feature/raw/*.csv` files (one `value` column each):
- `brand_keyword.csv`, `suspicious_tld.csv`, `free_hosting.csv`, `non_standard_port.csv`, `sorted_url.csv`, `auto_download_params.csv`

If these files are missing, hardcoded defaults are used. Run `data-migrate-feature` to populate them.

### Class Labels

All labels are normalized to binary: `benign` or `malicious`. Numeric labels (0=benign, 1/2/3=malicious) and string variants (phishing, defacement, spam, etc.) are all mapped via `data_dict.yaml`.
