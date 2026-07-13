# Operations

Diagrams and workflows reflecting the **current implementation** (as opposed to `docs/architecture.md`, which
records the target/aspirational structure from the in-progress migration). Cross-checked against
`src/tracking/model_registry.py`, `src/ml/prediction_service.py`, `src/ml/training_service.py`,
`src/cli/commands/model_registry.py`, and `src/workers/queue_worker.py`.

## System component architecture

```mermaid
flowchart LR
    CLI["CLI (cli/main.py)"] --> TrainingService
    CLI --> PredictionService
    CLI --> ModelRegistryManager
    Worker["Redis Worker (workers/queue_worker.py)"] --> TrainingService
    Worker --> PredictionService

    TrainingService --> DatasetPipeline
    TrainingService --> MLPipeline
    TrainingService --> MLflowTracker

    DatasetPipeline --> Validators["semd_ml.data.validators / splitters / versioning"]
    DatasetPipeline --> Extractor["semd_ml.features.extractor / url_normalizer / schema"]

    PredictionService --> CachedChampionModelLoader
    CachedChampionModelLoader --> ModelRegistryManager
    ModelRegistryManager --> MLPipeline

    MLflowTracker --> MLflowServer[(MLflow Tracking + Registry)]
    ModelRegistryManager --> MLflowServer
    TrainingService --> ArtifactsFS[(../models, ../reports, ../artifacts/mlflow)]
    Worker --> Redis[(Redis queues)]
    CLI --> Redis
```

## Dataset lifecycle

```mermaid
flowchart TD
    A["dataset/store/ archives"] --> B["data-migrate: extract to dataset/raw/"]
    B --> C["DatasetPipeline.load_and_merge_datasets"]
    C --> D["DatasetValidator.validate — stats, errors, warnings"]
    D --> E["DatasetValidator.clean — drop invalid URLs, duplicates, conflicting labels"]
    E --> F["compute_dataset_hash + build_dataset_metadata"]
    F --> G["extract_features (URLFeatureExtractor, parallel)"]
    G --> H["DatasetSplitter.split — group-aware train/test by registered_domain"]
    H --> I["Validation split (train_val_split) from train only"]
    I --> J["detect_imbalance + optional balancing (SMOTE / over / under)"]
    J --> K["dataset dict: X_train/X_val/X_test, feature_schema, dataset_metadata"]
```

Key invariants enforced by this pipeline (see `tests/unit/test_dataset_pipeline.py`):
- splitting groups by `registered_domain` so the same domain never appears in both train and test,
- balancing is applied to the training split only — validation/test stay at natural class ratios,
- the dataset hash is order-independent (sorted by `normalized_url`/`label`/`source` before hashing).

## Training workflow

```mermaid
flowchart TD
    A["CLI train / ml_training_queue job"] --> B[TrainingService.execute_training]
    B --> C["DatasetPipeline.prepare_dataset"]
    C --> D["MLPipeline.feature_selection (correlation -> variance -> mutual info -> SelectKBest)"]
    D --> E["MLPipeline.train_models — RandomizedSearchCV per algorithm"]
    E --> F["Evaluate on held-out test split — metrics, confusion matrix, ROC/PR curves"]
    F --> G["Pick best model by malicious_f1"]
    G --> H["MLPipeline.save_artifacts — .joblib + feature_schema.json"]
    H --> I["MLflowTracker — log params/metrics/artifacts, start/end run"]
    I --> J["result: tracking_run_id, best_artifact_path, per-algorithm metrics"]
```

`model_factory` (see `src/ml/model_factory.py`) is the single source of truth for supported algorithm identifiers
(`svm`, `random_forest`, `gradient_boosting`, and `xgboost` when the optional dependency is installed) — `decision_tree`
was retired from the identifier set.

## MLflow lifecycle

```mermaid
flowchart TD
    A[TrainingService starts a run] --> B["MLflowTracker.start_run"]
    B --> C["log params: algorithm, dataset_version, dataset_hash, feature_schema_version, balancing_method, python_version"]
    C --> D["log metrics: train/validation/test accuracy, malicious_f1, cross_validation_mean/std, ..."]
    D --> E["log artifacts: model .joblib, feature_schema.json, dataset_metadata.json, dataset_quality_report.json, sample_predictions.json, plots, requirements.txt"]
    E --> F["end_run"]
    F --> G["uv run semd-ml register --run-id <id> (ModelRegistryManager.register_candidate)"]
    G --> H["MLflow model version created, alias 'candidate' assigned"]
```

If MLflow is unavailable at training time, `MLflowTracker.start_run` fails gracefully: `execute_training` still
returns `status: success` with `tracking_run_id: null` and `tracking.enabled: false` (see
`test_mlflow_unavailable_does_not_break_training`) — training never blocks on tracking availability.

## Candidate promotion

```mermaid
flowchart TD
    A["uv run semd-ml promote [--model-version V]"] --> B["ModelRegistryManager.promote_candidate"]
    B --> C["_get_reference: candidate alias, or explicit version"]
    C --> D["_validate_reference"]
    D --> E["_validate_feature_schema: candidate schema_version == runtime schema_version"]
    D --> F["_validate_dataset_metadata: required keys present"]
    D --> G["_evaluate_gates: MODEL_PROMOTION_GATES metrics vs thresholds"]
    D --> H["_compare_to_champion: candidate must not regress vs current champion (if PROMOTION_REQUIRE_CHAMPION_COMPARISON)"]
    D --> I["_run_smoke_tests: load candidate, predict sample + configured smoke URLs"]
    E & F & G & H & I --> J{"all gates + comparison + smoke tests passed?"}
    J -- no --> K["raise ModelValidationError — promotion aborted"]
    J -- yes --> L["current champion (if any) -> alias 'previous-champion'"]
    L --> M["candidate version -> alias 'champion'"]
    M --> N["tag model version: promoted_at, promotion_status, validation_summary"]
```

Accuracy alone is never a promotion criterion — gates are `malicious_recall`, `malicious_f1`,
`false_negative_rate`, `prediction_latency_ms` by default (`MODEL_PROMOTION_GATES`).

## Inference flow

```mermaid
flowchart TD
    A["CLI predict / ml_prediction_queue job"] --> B["PredictionService.execute_prediction"]
    B --> C{"model_id in job_data?"}
    C -- yes, differs from cached --> D["CachedChampionModelLoader.load(selector=model_id)"]
    C -- no, no model cached yet --> E["CachedChampionModelLoader.load(selector='champion')"]
    C -- no, model already cached --> F["reuse cached pipeline"]
    D --> G["ModelRegistryManager.load_reference — download artifact, verify feature schema version"]
    E --> G
    G --> H{"registry reachable?"}
    H -- no, local fallback enabled --> I["load MLFLOW_LOCAL_FALLBACK_MODEL_PATH — alias 'local-fallback'"]
    H -- no, fallback disabled --> J["raise ModelRegistryError"]
    H -- yes --> K["cache pipeline+reference if alias == champion"]
    K --> L["normalize_url -> extract features -> align to schema -> model.predict"]
    F --> L
    I --> L
    L --> M["response: prediction, is_malicious, confidence, model_name, model_version, model_alias, feature_schema_version, prediction_time_ms"]
```

## Container startup

```bash
podman network create semd-shared-network   # once, before any compose file
make start                                  # docker/docker-compose.yml: ml-service + mlflow
make status
make logs LOG_SERVICE=mlflow
```

Startup order matters: `ml-service` has `depends_on: mlflow: condition: service_healthy`, so the MLflow container's
healthcheck (`curl -f http://localhost:5000`) must pass before the ML worker starts. If MLflow never reports healthy,
check `docs/troubleshooting.md`.
