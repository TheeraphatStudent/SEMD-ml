# Graph Report - .  (2026-07-17)

## Corpus Check
- 105 files · ~82,787 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 729 nodes · 1595 edges · 35 communities (28 shown, 7 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 211 edges (avg confidence: 0.7)
- Token cost: 289,029 input · 0 output

## Community Hubs (Navigation)
- Feature Extraction Engine
- Model Registry & Serving
- Dataset Pipeline
- ML Pipeline Core
- Configuration Settings
- CLI Command Handlers
- MLflow Tracker
- Queue Worker & Infra Validation
- Redis Client & Infra Fixes
- Refactoring Plan Tasks
- Training Service
- Import Verification Script
- Architecture Design Decisions
- Monitoring Metrics
- Monitoring Store
- Retraining & Rollback Docs
- Training Pipeline & Feature Config
- Archive Extraction Utils
- Investigation Findings & Fixes
- Monitoring Commands & Prediction Service
- Prediction Service Tests
- Monitoring & Prediction Tests
- Container Path Verification
- Database Client
- Retraining Dataset Tests
- Monitoring Store Tests
- CLI Bootstrap Tests
- Docker & MLflow Docs
- Legacy Feature Extractor
- Queue Manager
- JSON Reporting Utils
- Package Root

## God Nodes (most connected - your core abstractions)
1. `MLPipeline` - 44 edges
2. `ModelRegistryManager` - 42 edges
3. `MLflowTracker` - 38 edges
4. `URLFeatureExtractor` - 35 edges
5. `DatasetPipeline` - 34 edges
6. `emit_result()` - 29 edges
7. `MLServiceSettings` - 25 edges
8. `FakeMlflowClient` - 23 edges
9. `TrainingService` - 22 edges
10. `FeatureSchema` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Target features/schema.py module` --semantically_similar_to--> `FeatureExtractor`  [INFERRED] [semantically similar]
  docs/feature-schema.md → src/features/feature_extractor.py
- `Model Registry Doc` --semantically_similar_to--> `ModelRegistryManager`  [INFERRED] [semantically similar]
  docs/model-registry.md → src/tracking/model_registry.py
- `Manual retraining workflow (feedback->promotion)` --shares_data_with--> `MonitoringStore`  [INFERRED]
  docs/retraining.md → src/monitoring/store.py
- `Queue Worker Signal Handler Ordering Fix (D6)` --rationale_for--> `QueueWorker`  [INFERRED]
  docs/makefile-test-report.md → src/workers/queue_worker.py
- `Redis Client Socket Timeout Fix (D7)` --rationale_for--> `RedisClient`  [INFERRED]
  docs/makefile-test-report.md → src/infra/redis_client.py

## Import Cycles
- 1-file cycle: `src/cli/commands/data.py -> src/cli/commands/data.py`

## Hyperedges (group relationships)
- **Model Lifecycle Documentation Set (Registry, Serving, Tracking, Evaluation)** — docs_model_registry_doc, docs_model_serving_doc, docs_mlflow_tracking_doc, docs_model_evaluation_doc [INFERRED 0.85]
- **semd-ml Refactor & Validation Sequence** — docs_investigation_report_doc, docs_architecture_doc, docs_final_handoff_doc, docs_makefile_test_report_doc [INFERRED 0.85]
- **Training Orchestration Flow (CLAUDE.md Key Data Flow)** — src_data_dataset_pipeline_datasetpipeline, src_ml_ml_pipeline_mlpipeline, src_ml_training_service_trainingservice, src_tracking_mlflow_tracker_mlflowtracker [EXTRACTED 1.00]
- **Training workflow components (dataset -> feature selection -> training -> tracking -> registry)** — src_ml_training_service_trainingservice, src_data_dataset_pipeline_datasetpipeline, src_ml_ml_pipeline_mlpipeline, src_tracking_mlflow_tracker_mlflowtracker, src_tracking_model_registry_modelregistrymanager [INFERRED 0.85]
- **Manual feedback-to-promotion CLI workflow** — docs_retraining_workflow, docs_retraining_retrain_command, docs_retraining_register_command, docs_retraining_gate_check_command, docs_retraining_promote_command [EXTRACTED 1.00]
- **Session 2 resolved infrastructure blockers (T-092, T-093, Phase 4 queue-loss)** — docs_section_10_infrastructure_validation_t092, docs_section_10_infrastructure_validation_t093, docs_section_10_infrastructure_validation_phase4 [EXTRACTED 1.00]

## Communities (35 total, 7 thin omitted)

### Community 0 - "Feature Extraction Engine"
Cohesion: 0.05
Nodes (20): Any, Path, URLFeatureExtractor, build_feature_schema(), _expected_range_for_type(), FeatureSchema, FeatureSpec, Any (+12 more)

### Community 1 - "Model Registry & Serving"
Cohesion: 0.07
Nodes (16): MLflow Autologging Disabled Decision, CachedChampionModelLoader, GateResult, ModelReference, ModelRegistryError, ModelRegistryManager, ModelValidationError, Any (+8 more)

### Community 2 - "Dataset Pipeline"
Cohesion: 0.07
Nodes (18): DatasetPipeline, Any, DataFrame, Series, DatasetSplit, DatasetSplitter, DataFrame, Series (+10 more)

### Community 3 - "ML Pipeline Core"
Cohesion: 0.08
Nodes (14): ImbPipeline, ndarray, MLPipeline, Any, DataFrame, Series, TrainingArtifact, ModelDefinition (+6 more)

### Community 4 - "Configuration Settings"
Cohesion: 0.06
Nodes (14): BaseSettings, Logger, Config, FeaturesConfig, MLServiceSettings, Any, _xgboost_available(), get_logger() (+6 more)

### Community 5 - "CLI Command Handlers"
Cohesion: 0.13
Nodes (38): cmd_data_validate(), Any, cmd_evaluate(), Any, cmd_feature_engineering(), Any, cmd_data_migrate(), cmd_data_migrate_feature() (+30 more)

### Community 6 - "MLflow Tracker"
Cohesion: 0.08
Nodes (12): MLflowTracker, Any, RuntimeError, Raised when an existing MLflow experiment's artifact_location predates proxied s, UnsafeExperimentArtifactLocationError, MlflowArtifactPersistenceTests, MlflowExperimentReuseSafetyGuardTests, Regression coverage for T-093 (MLflow artifact persistence).  Requires a live ML (+4 more)

### Community 7 - "Queue Worker & Infra Validation"
Cohesion: 0.08
Nodes (18): Section 1/10 Infrastructure Validation doc, Phase 4: silent queue job loss fix, T-092: Redis auth alignment (backend<->ml-service), T-093: MLflow artifact persistence / experiment guard, T-094: Clean-environment regression coverage, _ensure_experiment, build_job_failure_result(), main() (+10 more)

### Community 8 - "Redis Client & Infra Fixes"
Cohesion: 0.17
Nodes (5): Redis Client Socket Timeout Fix (D7), Queue Worker Signal Handler Ordering Fix (D6), Makefile Verification Report, Any, RedisClient

### Community 9 - "Refactoring Plan Tasks"
Cohesion: 0.30
Nodes (16): Refactoring Plan doc, Adapter/compatibility-shim strategy, Task T01: package skeleton bootstrap, Task T02: split configuration into settings/paths/schemas, Task T03: extract dataset loading/validation/versioning, Task T04: URL normalization + feature schema abstractions, Task T05: fix data leakage (split before balancing), Task T06: decompose ml_pipeline into factory/training/eval/inference/artifacts (+8 more)

### Community 10 - "Training Service"
Cohesion: 0.31
Nodes (4): Any, Path, Series, TrainingService

### Community 12 - "Architecture Design Decisions"
Cohesion: 0.19
Nodes (15): Model Package Artifact Strategy, Split-Before-Balance Fix Policy, StratifiedGroupKFold Domain-Grouped Split, URL Normalization Policy, Target Architecture Doc, Dataset Pipeline Doc, Dataset Quality Report, Model Evaluation Doc (+7 more)

### Community 13 - "Monitoring Metrics"
Cohesion: 0.27
Nodes (7): compute_monitoring_metrics(), _empty_metrics(), _percentile(), Any, Aggregate prediction-event telemetry.      `estimated_false_positive_rate` / `es, _event(), MonitoringMetricsTests

### Community 14 - "Monitoring Store"
Cohesion: 0.23
Nodes (6): Connection, hash_url(), MonitoringStore, Any, Self-contained SQLite store for prediction telemetry, independent of MLflow runs, _utc_now()

### Community 15 - "Retraining & Rollback Docs"
Cohesion: 0.18
Nodes (13): Operations doc (current implementation diagrams), Retraining doc, uv run semd-ml gate-check, uv run semd-ml promote, uv run semd-ml register, uv run semd-ml retrain, Manual retraining workflow (feedback->promotion), Rollback doc (+5 more)

### Community 16 - "Training Pipeline & Feature Config"
Cohesion: 0.15
Nodes (12): Training Pipeline doc, Single .joblib inference artifact composition, Training pipeline 9-step flow, data_dict.yaml (column/label mapping config), dataset_feature.yaml (feature reference CSV migration config), features.yaml (feature schema config), class_feature_emphasis (benign/malicious weighting), domain_level feature group (22 features) (+4 more)

### Community 17 - "Archive Extraction Utils"
Cohesion: 0.42
Nodes (11): _ensure_within_directory(), extract_archive(), extract_csvs_from_archive(), find_archives(), is_supported_archive(), move_csvs_from_directory(), Any, Path (+3 more)

### Community 18 - "Investigation Findings & Fixes"
Cohesion: 0.29
Nodes (11): Alias-based Model Registry (candidate/champion/previous-champion), MLflow Artifact Root / Container CWD Mismatch Risk, CWD-Dependent Path Defect, Pre-split Balancing Data Leakage Defect, Local Fallback Serving Policy, Model Promotion Gates Policy, Final Handoff Section 8 Report, Investigation Report (+3 more)

### Community 19 - "Monitoring Commands & Prediction Service"
Cohesion: 0.29
Nodes (7): Prediction Monitoring doc, uv run semd-ml feedback, uv run semd-ml monitor, prediction_events table schema, uv run semd-ml review, PredictionService, Any

### Community 22 - "Container Path Verification"
Cohesion: 0.47
Nodes (9): check_artifact_root_uses_proxied_scheme(), check_ml_service_and_mlflow_share_artifact_host_dir(), check_mlflow_healthcheck_does_not_use_curl(), check_mlflow_image_matches_client(), check_redis_env_forwarded(), fail(), load_compose(), main() (+1 more)

### Community 24 - "Retraining Dataset Tests"
Cohesion: 0.29
Nodes (4): build_feedback_dataset(), Any, Turn admin-approved prediction feedback into a labeled CSV for retraining., RetrainingDatasetTests

### Community 26 - "CLI Bootstrap Tests"
Cohesion: 0.33
Nodes (4): CompletedProcess, CliBootstrapTests, Clean-environment regression coverage for the CLI import chain and entrypoints., run_in_src()

### Community 27 - "Docker & MLflow Docs"
Cohesion: 0.33
Nodes (9): ML Service Docker Compose, mlflow container, ml-service container, semd-shared-network, Feature Schema Doc, MLflow Tracking Doc, SEMD-ML README, Target features/extractor.py module (+1 more)

### Community 29 - "Queue Manager"
Cohesion: 0.32
Nodes (3): SEMD-ML CLAUDE.md Guide, Any, QueueManager

### Community 30 - "JSON Reporting Utils"
Cohesion: 0.53
Nodes (5): emit_json_result(), normalize_json_filename(), Any, Path, write_json_result()

## Knowledge Gaps
- **30 isolated node(s):** `semd-ml`, `Config`, `semd-shared-network`, `Target config/settings.py module`, `Model Package Manifest schema` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelRegistryManager` connect `Model Registry & Serving` to `Feature Extraction Engine`, `ML Pipeline Core`, `CLI Command Handlers`, `MLflow Tracker`, `Retraining & Rollback Docs`, `Investigation Findings & Fixes`?**
  _High betweenness centrality (0.154) - this node is a cross-community bridge._
- **Why does `DatasetPipeline` connect `Dataset Pipeline` to `Model Registry & Serving`, `ML Pipeline Core`, `Architecture Design Decisions`, `Retraining & Rollback Docs`, `Investigation Findings & Fixes`, `Queue Manager`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Why does `SEMD-ML CLAUDE.md Guide` connect `Queue Manager` to `Dataset Pipeline`, `ML Pipeline Core`, `Configuration Settings`, `MLflow Tracker`, `Queue Worker & Infra Validation`, `Redis Client & Infra Fixes`, `Training Service`, `Monitoring Commands & Prediction Service`, `Database Client`, `Docker & MLflow Docs`, `Legacy Feature Extractor`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MLPipeline` (e.g. with `FeatureSchema` and `CachedChampionModelLoader`) actually correct?**
  _`MLPipeline` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `ModelRegistryManager` (e.g. with `Model Registry Doc` and `cmd_gate_check()`) actually correct?**
  _`ModelRegistryManager` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `MLflowTracker` (e.g. with `ml-service container` and `ModelRegistryManager`) actually correct?**
  _`MLflowTracker` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `URLFeatureExtractor` (e.g. with `FeatureExtractor` and `.__init__()`) actually correct?**
  _`URLFeatureExtractor` has 9 INFERRED edges - model-reasoned connections that need verification._