# Investigation Report

Date: 2026-07-14
Project: `semd-ml`

## Current architecture

`semd-ml` is a Python ML service for malicious URL classification. The codebase is organized around a CLI-first workflow under `src/`, with shared global singletons created at import time:

- `core.config`: runtime settings and feature config loading
- `features.feature_extractor`: URL feature extraction
- `data.dataset_pipeline`: dataset loading, normalization, balancing, splitting
- `ml.ml_pipeline`: preprocessing, search/training, evaluation, artifact save/load
- `ml.training_service`: orchestration for normal and one-by-one training
- `ml.prediction_service`: inference and database writeback
- `tracking.mlflow_tracker`: MLflow run/model logging
- `infra.database` / `infra.redis_client`: PostgreSQL and Redis clients
- `queues.queue_manager` / `workers.queue_worker`: async queue integration
- `cli.main` and `cli.commands.*`: user entry points

The service assumes execution from `src/`. Many paths are relative to that working directory.

## Directory tree

```text
semd-ml/
├── README.md
├── CLAUDE.md
├── makefile
├── requirements.txt
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   └── investigation-report.md
├── mlflow_data/
│   └── mlflow.db
├── models/
│   ├── xgboost_model_dffd5ea189914e658a243740064e956f.pkl
│   ├── xgboost_scaler_dffd5ea189914e658a243740064e956f.pkl
│   └── xgboost_label_encoder_dffd5ea189914e658a243740064e956f.pkl
├── reports/
│   └── many generated training/evaluation/prediction JSON and CSV artifacts
└── src/
    ├── main.py
    ├── verify_imports.py
    ├── cli/
    │   ├── main.py
    │   ├── common.py
    │   └── commands/
    ├── core/
    │   ├── config.py
    │   ├── logger.py
    │   ├── archive_utils.py
    │   └── reporting.py
    ├── data/
    │   ├── data_dict.yaml
    │   └── dataset_pipeline.py
    ├── dataset/
    │   ├── raw/
    │   ├── extraction/
    │   ├── feature/
    │   │   ├── dataset_feature.yaml
    │   │   ├── raw/
    │   │   └── store/
    │   ├── store/
    │   ├── script/
    │   └── test/
    ├── features/
    │   ├── features.yaml
    │   └── feature_extractor.py
    ├── infra/
    │   ├── database.py
    │   └── redis_client.py
    ├── ml/
    │   ├── ml_pipeline.py
    │   ├── training_service.py
    │   └── prediction_service.py
    ├── queues/
    │   └── queue_manager.py
    ├── tracking/
    │   └── mlflow_tracker.py
    └── workers/
        └── queue_worker.py
```

## Package structure

- The effective Python package root is `src/`.
- Public package init files re-export singletons and classes.
- `src/main.py` delegates to `cli.main.main()`.
- `src/cli/main.py` defines the subcommands:
  - `train`
  - `train-obo`
  - `predict`
  - `predict-test`
  - `evaluate`
  - `feature-engineering`
  - `worker`
  - `queue-status`
  - `data-migrate`
  - `data-migrate-feature`

## Configuration and environment

- Dependency file: `requirements.txt`
- No `pyproject.toml`, `setup.py`, `uv.lock`, `ruff.toml`, `mypy.ini`, or `pytest.ini` were found.
- Environment files are under `src/`:
  - `src/.env`
  - `src/.env.example`
- Container files:
  - `docker/Dockerfile`
  - `docker/docker-compose.yml`
- Existing model artifact family:
  - one XGBoost model/scaler/label encoder triplet in `models/`
- Existing MLflow artifact store:
  - `mlflow_data/mlflow.db`

## Existing documentation

- `README.md` describes the intended structure, CLI, queues, and Docker usage.
- `CLAUDE.md` gives a concise module map and data flow.
- Both documents overstate the presence of feature selection in the active training path.

## Current workflows

### Dataset ingestion flow

1. CLI `train` or `evaluate` calls `dataset_pipeline.prepare_dataset(...)`.
2. `DatasetPipeline.load_and_merge_datasets()` loads CSV/XLSX data from `src/dataset/raw`.
3. Column names are standardized via `src/data/data_dict.yaml`.
4. Labels are normalized to binary `benign` / `malicious`.
5. Conflicting duplicate URLs are dropped entirely.
6. A cached `merged.csv` is written to `src/dataset/raw/merged.csv`.

### Data-cleaning flow

1. `preprocess_dataset()` drops duplicate URLs.
2. Missing URLs are removed.
3. Labels are lowercased and normalized.
4. Rows with labels outside the configured class set are discarded.

### Feature-extraction flow

1. `feature_extractor.extract(url)` computes URL, domain, path, query, and sequence features.
2. Lookup tables are loaded from `src/dataset/feature/raw/*.csv`, with hardcoded defaults as fallback.
3. `DatasetPipeline.extract_features()` parallelizes extraction with `multiprocessing.Pool`.
4. Extracted features are saved to:
   - `src/dataset/extraction/extracted_features.csv`
   - `src/dataset/extraction/features_before_balance.csv`

### Training flow

1. `cli.commands.train.cmd_train()` optionally runs feature-reference migration.
2. `TrainingService.execute_training()` prepares the dataset.
3. `MLflowTracker.start_run()` opens a run.
4. `MLPipeline.preprocess_data()` label-encodes labels and standard-scales features.
5. `TrainingService` explicitly skips feature selection and uses all features.
6. `MLPipeline.train_and_compare_models()` runs `RandomizedSearchCV` per algorithm.
7. Best model is selected by test-set weighted F1.
8. `MLPipeline.save_artifacts()` saves model, scaler, and label encoder to `models/`.
9. Reports are written to `reports/`.
10. Optional DB update writes model registry metadata.

### Cross-validation flow

- Implemented inside `MLPipeline.train_model()` with `RandomizedSearchCV(cv=settings.cv_folds)`.
- Default `cv_folds` is `5`.
- No separate holdout-validation abstraction exists beyond train/test split plus CV inside the training partition.

### Evaluation flow

- `MLPipeline.evaluate_model()` computes accuracy, precision, recall, F1, confusion matrix, and classification report on `X_test`.
- `cli.commands.evaluate.cmd_evaluate()` retrains models and emits an evaluation summary.
- `MLflowTracker.evaluate_model()` logs evaluation metrics and a text classification report.

### Model-saving flow

- Standard training saves:
  - `{algorithm}_model_{run_id}.pkl`
  - `{algorithm}_scaler_{run_id}.pkl`
  - `{algorithm}_label_encoder_{run_id}.pkl`
- One-by-one training saves nested artifacts under `models/<clean_name>/...`.
- MLflow logs artifacts and attempts model registry registration.

### Model-loading flow

- `PredictionService.load_model(run_id)` calls `ml_pipeline.load_artifacts(run_id)`.
- Artifact lookup is filename-based and only searches directly under `models/`.

### Prediction flow

1. CLI or queue job passes URL and optional `model_id`.
2. `PredictionService` loads artifacts when needed.
3. `MLPipeline.predict()` extracts features from the URL.
4. Features are reindexed to match scaler input columns when possible.
5. Scaler transform is applied.
6. Saved model predicts class and probabilities.
7. Result is returned, with optional DB persistence.

## Training sequence

```text
CLI train
-> cmd_train
-> TrainingService.execute_training
-> DatasetPipeline.prepare_dataset
-> load_and_merge_datasets
-> preprocess_dataset
-> extract_features
-> detect_imbalance
-> apply_balancing
-> split_dataset
-> MLflowTracker.start_run
-> MLPipeline.preprocess_data
-> MLPipeline.train_and_compare_models
-> MLPipeline.evaluate_model
-> MLPipeline.save_artifacts
-> MLflow model/artifact logging
-> optional PostgreSQL model registry update
```

## Prediction sequence

```text
CLI predict / queue job
-> PredictionService.execute_prediction
-> load_model(run_id) if needed
-> MLPipeline.load_artifacts
-> MLPipeline.predict
-> FeatureExtractor.extract
-> scaler transform
-> model.predict / predict_proba
-> optional PostgreSQL prediction insert
```

## Integration points

### Backend integration

- PostgreSQL via `infra.database.DatabaseClient`
- Redis via `infra.redis_client.RedisClient`
- Async worker via `workers.queue_worker.QueueWorker`
- Queue names:
  - `ml_training_queue`
  - `ml_prediction_queue`
  - `ml_result_queue`

### Existing MLflow integration

- Tracking URI comes from `MLFLOW_TRACKING_URI`.
- `MLflowTracker` creates or reuses experiment `malicious_url_detection`.
- Logs params, metrics, artifacts, text reports, and sklearn models.
- Attempts alias assignment and model version stage transition.
- Local MLflow backend store is SQLite at `mlflow_data/mlflow.db`.

## Existing test status

- No `tests/` or `test_*.py` suite exists.
- Only sample CSV files exist under `src/dataset/test/`.
- The repository includes `src/verify_imports.py`, but it failed immediately because required dependencies are not installed in the current environment.

## Commands executed

### Requested commands

| Command | Result |
|---|---|
| `uv --version` | Failed: `/bin/bash: uv: command not found` |
| `uv sync` | Failed: `/bin/bash: uv: command not found` |
| `uv run pytest` | Failed: `/bin/bash: uv: command not found` |
| `uv run ruff check .` | Failed: `/bin/bash: uv: command not found` |
| `uv run ruff format --check .` | Failed: `/bin/bash: uv: command not found` |
| `uv run mypy src` | Failed: `/bin/bash: uv: command not found` |

### Equivalent fallback checks attempted

| Command | Result |
|---|---|
| `python3 --version` | Passed: `Python 3.12.3` |
| `python3 -m pytest` | Failed: `No module named pytest` |
| `python3 verify_imports.py` (from `src/`) | Failed: `No module named pydantic_settings` |
| `.venv/bin/python --version` | Failed: `.venv/bin/python: No such file or directory` |

### Tooling observations

- `uv` is not installed in the environment.
- The checked-in `.venv` is broken: `.venv/bin/python` points to `/usr/bin/python`, which does not exist here.
- `pytest`, `ruff`, `mypy`, and even `pip` are not available from the active system interpreter.

## Issue list

### Critical

| Severity | File path | Class/function | Problem | Impact | Recommended fix |
|---|---|---|---|---|---|
| Critical | `src/data/dataset_pipeline.py:418-538` | `DatasetPipeline.apply_balancing`, `prepare_dataset` | Resampling happens before `train_test_split`. SMOTE/over/under-sampling therefore affects the eventual test set. | Evaluation metrics are biased and cannot be trusted; synthetic or duplicated samples leak into test data. | Split first, then fit balancing only on `X_train`/`y_train`. Keep `X_test` untouched. |
| Critical | `docker/Dockerfile:3-20`, `src/core/config.py:40-45`, `src/main.py:1-9` | container entrypoint / settings | Container `CMD ["python", "main.py", ...]` runs from `/app`, but `main.py` lives in `/app/src`, and runtime paths assume the working directory is `src/`. | The containerized service is very likely non-starting or misconfigured even before training/prediction begin. | Set `WORKDIR /app/src` or use `CMD ["python", "src/main.py", ...]`, and convert runtime paths to be file-relative rather than cwd-relative. |
| Critical | `src/core/config.py:124-130` | `FeaturesConfig._load_config` | Feature config is loaded from a raw relative path (`./features/features.yaml`) at import time. | Any execution outside `src/` fails during import, including container, scripts, and external package usage. | Resolve config paths relative to the module file or repository root, not the process cwd. |

### High

| Severity | File path | Class/function | Problem | Impact | Recommended fix |
|---|---|---|---|---|---|
| High | `src/data/dataset_pipeline.py:118-146` | `DatasetPipeline.load_and_merge_datasets` | All dataset selections share a single cache file `merged.csv`, and cache reuse is based only on modification times of present source files. | Training on one subset can silently reuse merged data from a different subset, producing wrong experiments and reports. | Key the cache by the sorted input file list and/or disable cache reuse unless the dataset selection exactly matches. |
| High | `src/ml/training_service.py:556-585`, `src/ml/ml_pipeline.py:352-365` | `_save_artifacts_obo`, `_find_artifact` | One-by-one artifacts are saved in nested directories, but loader only searches the top-level `models/` directory. | Models produced by `train-obo` are not loadable by the current prediction path. | Store a registry mapping or make artifact discovery recursive and dataset-aware. |
| High | `src/ml/prediction_service.py:81-102` | `PredictionService.execute_prediction` | Database lookup parses `current_model_id` as an integer suffix, but normal run IDs are opaque strings. The exception is swallowed. | Prediction persistence to PostgreSQL will usually fail silently, weakening backend integration and auditability. | Store/load by MLflow run ID directly, or persist a separate numeric model registry ID in the job payload. |
| High | `src/__init__.py:1-35` | package init | Imports `dataset.store.cloudflare_client` and `dataset.store.hugging_face`, but the actual scripts are under `src/dataset/script/`. | `import src` or any code relying on the umbrella package export can fail immediately. | Remove these exports or point them to the correct modules. |
| High | `src/ml/training_service.py:77-81`, `README.md`, `CLAUDE.md` | `TrainingService.execute_training` | The active training path explicitly says “Using all features (no feature selection)” even though documentation claims a multi-stage feature selection pipeline. | Architecture docs are misleading; downstream refactoring based on docs will be wrong. | Either implement the documented feature-selection stages or update docs and naming to match the real pipeline. |

### Medium

| Severity | File path | Class/function | Problem | Impact | Recommended fix |
|---|---|---|---|---|---|
| Medium | `src/queues/queue_manager.py:32-47`, `src/infra/redis_client.py:17-22` | `QueueManager.get_queue_status` | Redis client uses `decode_responses=True`, but queue status still calls `job.decode('utf-8')`. | Queue inspection will emit parse errors for valid jobs and hide real queue state. | Remove `.decode(...)` and parse the returned string directly. |
| Medium | `src/ml/prediction_service.py:164-191` | `_generate_suggestion` | Response text assumes multiclass outputs like `phishing`, `malware`, `redirect`, `spam`, but the pipeline normalizes everything to binary `benign`/`malicious`. | Prediction semantics exposed to users/backend are inconsistent with the trained label space. | Align messaging with binary output or preserve a real multiclass label space end-to-end. |
| Medium | `src/ml/ml_pipeline.py:52-72`, `src/ml/ml_pipeline.py:94-97` | `preprocess_data`, `train_model` | Features are scaled once before training and then again inside a sklearn `Pipeline` with another `StandardScaler`. | This is redundant, complicates artifact semantics, and makes training/inference harder to reason about. | Keep scaling in one place only, preferably inside the sklearn pipeline. |
| Medium | `docker/docker-compose.yml:12-17` | service config | Compose sets `POSTGRES_HOST=postgreSQL` and `POSTGRES_DB=semd`, while the repo defaults and docs use other names; no matching DB service is defined in this file. | Backend/service startup depends on external naming conventions and is fragile across environments. | Centralize service hostnames, align names with the actual backend compose, and document the contract once. |
| Medium | `src/features/feature_extractor.py:52-67` | `_load_feature_values` | Broad exception suppression hides CSV/schema problems and silently falls back to defaults. | Feature semantics can drift without any visible failure, especially for migrated feature reference files. | Log validation failures with filename/schema detail and fail fast in non-development environments. |

### Low

| Severity | File path | Class/function | Problem | Impact | Recommended fix |
|---|---|---|---|---|---|
| Low | `requirements.txt` | dependency manifest | The repo has no dev dependency set for `pytest`, `ruff`, or `mypy`, despite expecting those checks to run. | Basic quality checks are not reproducible from the repo alone. | Add a proper project manifest or dev requirements file with the expected tooling. |
| Low | `src/verify_imports.py` | `ImportVerifier` | Smoke test exists, but it is not integrated into CI and depends on the environment being manually provisioned. | Import regressions are likely to slip through. | Add a real automated test suite and run import smoke tests in CI. |
| Low | `src/ml/ml_pipeline.py:117-138` | `train_model` | Prints large training output directly to stdout and writes best-result CSVs unconditionally. | Noisy CLI behavior and harder automation/log parsing. | Route summaries through structured logging and make report writing explicit. |

## Risks

- Current evaluation metrics are likely optimistic because of pre-split balancing.
- Container startup appears broken in the default image.
- OBO models are operationally disconnected from the prediction path.
- Queue monitoring and prediction persistence both have hidden failure modes.
- The environment is not reproducible from the repository in its current state.

## Refactoring recommendations

1. Make path handling deterministic.
   Convert all runtime paths to module-relative or config-driven absolute paths.

2. Fix data leakage before any architectural expansion.
   Split raw features into train/test first, then apply balancing only to training data.

3. Unify model preprocessing.
   Keep feature order, scaling, and metadata inside one persisted inference artifact.

4. Separate artifact identity from file naming.
   Persist a manifest with run ID, algorithm, feature names, scaler/model paths, and training mode.

5. Formalize the project toolchain.
   Add `pyproject.toml` or an equivalent manifest, declare dev tools, and make `uv sync` actually usable.

6. Replace smoke testing with real tests.
   Add unit tests for label normalization, feature extraction, dataset merging, artifact loading, and prediction service integration.

## Open questions

1. Is the intended label space binary (`benign` / `malicious`) or multiclass (`phishing`, `spam`, `redirect`, etc.)?
2. Should `train-obo` models be first-class prediction targets, or are they only for offline comparison?
3. Is feature selection supposed to exist in production, or has the design changed and docs were never updated?
4. Which artifact source is authoritative for inference: local `models/` files, MLflow registry, or PostgreSQL model registry?
5. Should the service be runnable from the repo root, from `src/`, and from Docker with the same path semantics?

# Session Handoff

## Completed
- Inspected the repository structure, source modules, CLI, config files, container files, datasets, model artifacts, reports, and documentation.
- Reconstructed the current ingestion, feature extraction, training, evaluation, artifact, prediction, queue, database, and MLflow workflows.
- Executed the requested verification commands and recorded exact failures.
- Produced a ranked issue list with file paths, functions, impact, and recommended fixes.
- Created this investigation report.

## Important findings
- The training pipeline leaks resampled data into the test set.
- The Docker entrypoint and cwd-dependent config paths are inconsistent and likely break container startup.
- `train-obo` artifacts are not loadable by the current prediction loader.
- Prediction DB persistence and queue status inspection both have hidden integration bugs.
- The repo cannot currently reproduce the requested quality checks because `uv` and dev tooling are missing or broken.

## Files inspected
- `README.md`
- `CLAUDE.md`
- `makefile`
- `requirements.txt`
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `src/main.py`
- `src/verify_imports.py`
- `src/core/*.py`
- `src/data/data_dict.yaml`
- `src/data/dataset_pipeline.py`
- `src/features/features.yaml`
- `src/features/feature_extractor.py`
- `src/ml/*.py`
- `src/tracking/mlflow_tracker.py`
- `src/infra/*.py`
- `src/queues/queue_manager.py`
- `src/workers/queue_worker.py`
- `src/cli/main.py`
- `src/cli/common.py`
- `src/cli/commands/*.py`
- `src/.env`
- `src/.env.example`

## Commands executed
- `find /home/semd/Desktop/Project/SEMD/semd-ml -maxdepth 3 ...`
- `rg --files /home/semd/Desktop/Project/SEMD/semd-ml`
- `sed -n ...` across repo source and config files
- `rg -n ...` across repo source files
- `uv --version`
- `uv sync`
- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `python3 --version`
- `python3 -m pytest`
- `python3 verify_imports.py`

## Failed commands
- `uv --version`
- `uv sync`
- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `.venv/bin/python --version`
- `python3 -m pytest`
- `python3 verify_imports.py`

## Critical risks
- Invalid model evaluation due to dataset leakage.
- Non-functional or unstable container runtime due to cwd/path assumptions.
- Inference artifact loading gaps between standard and OBO training modes.

## Recommended next section
Section 2 — Architecture and Refactoring Plan
