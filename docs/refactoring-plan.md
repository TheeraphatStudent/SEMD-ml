# Refactoring Plan

Date: 2026-07-14
Project: `semd-ml`

## Scope

This plan defines the migration path to the target architecture described in `docs/architecture.md`. It is grounded in the current CLI-first codebase and explicitly avoids a big-bang rewrite. The first phases focus on correctness and compatibility: path stability, leakage-free evaluation, artifact identity, and preserved service contracts.

## Compatibility strategy

### Existing public interfaces

Current public interfaces that are already exposed in code or documentation:

- CLI entrypoint: `src/main.py`
- CLI command names:
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
- Package exports in `src/__init__.py`
- Service methods:
  - `TrainingService.execute_training(job_data)`
  - `TrainingService.execute_training_obo(job_data)`
  - `PredictionService.execute_prediction(job_data)`
  - `PredictionService.batch_predict(job_data)`
- Artifact naming convention under `models/`
- MLflow run IDs used as effective model IDs in current CLI usage

### Backend callers

Likely backend-facing dependencies from current code:

- PostgreSQL model registry update in training service
- PostgreSQL prediction persistence in prediction service
- Redis queues:
  - `ml_training_queue`
  - `ml_prediction_queue`
  - `ml_result_queue`
- Queue workers that deserialize the same job payload shapes as the CLI-backed services

### CLI callers

Current callers likely depend on:

- Existing command names and flags
- JSON output shape emitted by CLI commands
- Ability to run from `src/` today
- Model lookup via `--model-id <run_id>`

### Artifact consumers

Current artifact consumers likely include:

- `PredictionService.load_model(run_id)`
- Manual filesystem users reading `models/*.pkl`
- MLflow UI and registry users
- PostgreSQL model registry rows that store metadata derived from a training run

### Interfaces that must remain compatible

- CLI command names and core flags
- `job_data` payload keys accepted by training and prediction services
- `PredictionService.execute_prediction()` response shape:
  - `status`
  - `url`
  - `prediction.class`
  - `prediction.is_malicious`
  - `prediction.confidence`
  - `prediction.probabilities`
  - `model_id`
- Use of MLflow run ID as a valid model reference during transition
- Local-artifact inference path for existing trained models

### Interfaces that may require adapters

- `src/__init__.py` umbrella exports
- Internal imports like `from core import settings` and `from ml import ml_pipeline`
- File path assumptions tied to execution from `src/`
- OBO artifact discovery
- Database model registry lookups that currently assume numeric IDs
- Any future move from loose `.pkl` files to manifest-based model packages

### Adapter strategy

- Keep `src/main.py` and `src/cli/*` as thin wrappers around the new package.
- Introduce compatibility shims in existing module paths before deleting old modules.
- Make artifact loading accept both:
  - legacy triplet files
  - manifest-based model packages
- Allow `model_id` resolution from:
  - MLflow run ID
  - manifest model ID
  - optional registry alias

## Implementation plan

### Task T01

- Description: Introduce package skeleton `src/semd_ml/` and bootstrap wiring without changing behavior.
- Files affected:
  - `pyproject.toml`
  - `src/semd_ml/__init__.py`
  - `src/semd_ml/bootstrap.py`
  - `src/main.py`
  - `src/__init__.py`
- Dependencies: None
- Risk: Low
- Acceptance criteria:
  - Project imports through `semd_ml` without requiring execution from `src/`.
  - Existing `src/main.py` still runs current CLI commands.
  - Old import paths still work through compatibility wrappers.
- Required tests:
  - Import smoke test for `semd_ml`
  - CLI smoke test for `python src/main.py --help`

### Task T02

- Description: Split configuration into settings, path resolution, and config schema loaders.
- Files affected:
  - `src/core/config.py`
  - `src/semd_ml/config/settings.py`
  - `src/semd_ml/config/paths.py`
  - `src/semd_ml/config/schemas.py`
  - `.env.example`
  - `configs/app.yaml`
  - `configs/features.yaml`
  - `configs/data_dict.yaml`
- Dependencies: T01
- Risk: Medium
- Acceptance criteria:
  - All paths resolve correctly from repo root, `src/`, and container runtime.
  - Feature config and data dict loading no longer depend on cwd.
  - Existing env var names remain supported.
- Required tests:
  - Unit tests for path resolution
  - Integration test for loading settings from temp env/config files

### Task T03

- Description: Extract dataset loading, validation, cleaning, and versioning into dedicated modules.
- Files affected:
  - `src/data/dataset_pipeline.py`
  - `src/semd_ml/data/loaders.py`
  - `src/semd_ml/data/validators.py`
  - `src/semd_ml/data/cleaners.py`
  - `src/semd_ml/data/versioning.py`
  - `src/semd_ml/data/repositories.py`
- Dependencies: T02
- Risk: Medium
- Acceptance criteria:
  - Canonical dataset build returns the same columns as today plus provenance metadata.
  - Merged dataset cache is keyed by dataset selection and fingerprint, not a single global file.
  - Validation and cleaning statistics are exposed separately.
- Required tests:
  - Unit tests for label normalization
  - Unit tests for conflicting-label URL handling
  - Unit tests for dataset fingerprint generation

### Task T04

- Description: Introduce URL normalization and feature schema abstractions around the current feature extractor.
- Files affected:
  - `src/features/feature_extractor.py`
  - `src/semd_ml/features/url_normalizer.py`
  - `src/semd_ml/features/schema.py`
  - `src/semd_ml/features/reference_store.py`
  - `src/semd_ml/features/extractor.py`
- Dependencies: T02
- Risk: Medium
- Acceptance criteria:
  - Training and inference both call the same URL normalization code.
  - Feature ordering is centrally defined and serializable.
  - Reference CSV load failures become visible and diagnosable.
- Required tests:
  - Unit tests for URL normalization
  - Unit tests for schema alignment/default filling
  - Unit tests for reference-store validation behavior

### Task T05

- Description: Fix the data leakage by splitting before balancing and formalize the dataset build pipeline.
- Files affected:
  - `src/data/dataset_pipeline.py`
  - `src/semd_ml/data/splitters.py`
  - `src/semd_ml/pipelines/dataset_build_pipeline.py`
- Dependencies: T03, T04
- Risk: High
- Acceptance criteria:
  - Balancing is applied only to the training partition.
  - Test partition remains untouched and reproducible.
  - Reported dataset statistics distinguish raw, cleaned, split, and balanced counts.
- Required tests:
  - Unit test proving no synthetic examples enter the test set
  - Integration test for stratified split plus selected balancing method

### Task T06

- Description: Decompose `ml_pipeline` into model factory, training, evaluation, inference, and artifact modules.
- Files affected:
  - `src/ml/ml_pipeline.py`
  - `src/semd_ml/models/factory.py`
  - `src/semd_ml/models/training.py`
  - `src/semd_ml/models/evaluation.py`
  - `src/semd_ml/models/inference.py`
  - `src/semd_ml/models/artifacts.py`
  - `src/semd_ml/models/package_manifest.py`
- Dependencies: T05
- Risk: High
- Acceptance criteria:
  - Scaling occurs in one place only.
  - Training emits a manifest-based model package.
  - Inference can load both legacy artifacts and new model packages.
- Required tests:
  - Unit tests for algorithm factory output
  - Integration test for train-save-load-predict round trip
  - Regression test for legacy artifact loading

### Task T07

- Description: Separate training and prediction workflows into dedicated pipelines while preserving service entrypoints.
- Files affected:
  - `src/ml/training_service.py`
  - `src/ml/prediction_service.py`
  - `src/semd_ml/pipelines/training_pipeline.py`
  - `src/semd_ml/pipelines/prediction_pipeline.py`
  - `src/semd_ml/services/training_service.py`
  - `src/semd_ml/services/prediction_service.py`
- Dependencies: T05, T06
- Risk: Medium
- Acceptance criteria:
  - Existing job payloads still work.
  - Services become thin orchestration layers over pipelines.
  - Prediction result shape remains backward compatible.
- Required tests:
  - Integration tests for CLI-equivalent training job payload
  - Integration tests for single and batch prediction payloads

### Task T08

- Description: Refactor MLflow integration into tracking, registry, and promotion responsibilities.
- Files affected:
  - `src/tracking/mlflow_tracker.py`
  - `src/semd_ml/mlops/tracking.py`
  - `src/semd_ml/mlops/registry.py`
  - `src/semd_ml/mlops/promotion.py`
  - `src/semd_ml/mlops/lineage.py`
- Dependencies: T06, T07
- Risk: Medium
- Acceptance criteria:
  - Tracking run logging works independently of model promotion.
  - Model registration stores manifest metadata and dataset lineage.
  - Alias/stage transitions are policy-driven and optional.
- Required tests:
  - Unit tests with mocked MLflow client
  - Integration test for run logging and model registration metadata

### Task T09

- Description: Repair model identity and artifact lookup across standard and OBO training modes.
- Files affected:
  - `src/ml/training_service.py`
  - `src/ml/prediction_service.py`
  - `src/semd_ml/models/artifacts.py`
  - `src/semd_ml/mlops/registry.py`
  - `artifacts/models/`
- Dependencies: T06, T07, T08
- Risk: High
- Acceptance criteria:
  - OBO-trained models are loadable by the same prediction path.
  - `model_id` resolution works for run ID, manifest ID, and optional alias.
  - Artifact discovery is no longer limited to a single directory level.
- Required tests:
  - Integration test for loading an OBO-trained artifact
  - Regression test for current top-level model files

### Task T10

- Description: Stabilize database and queue integrations behind explicit adapters and fix hidden parsing bugs.
- Files affected:
  - `src/infra/database.py`
  - `src/infra/redis_client.py`
  - `src/queues/queue_manager.py`
  - `src/workers/queue_worker.py`
  - `src/semd_ml/infra/database.py`
  - `src/semd_ml/infra/redis.py`
  - `src/semd_ml/services/worker_service.py`
- Dependencies: T07
- Risk: Medium
- Acceptance criteria:
  - Queue status works with decoded Redis responses.
  - Prediction persistence uses a stable model reference instead of parsing a numeric suffix from run ID.
  - Worker behavior matches current queue contracts.
- Required tests:
  - Unit tests for queue status parsing
  - Integration tests with mocked DB and Redis clients

### Task T11

- Description: Preserve CLI compatibility while moving command implementations into the new package.
- Files affected:
  - `src/cli/main.py`
  - `src/cli/common.py`
  - `src/cli/commands/*.py`
  - `src/semd_ml/interfaces/cli/*`
- Dependencies: T07, T10
- Risk: Low
- Acceptance criteria:
  - All existing command names and main flags still work.
  - Output JSON contracts remain unchanged unless explicitly versioned.
  - CLI no longer relies on import-time singleton state.
- Required tests:
  - CLI integration tests for `train --help`, `predict --help`, `evaluate --help`
  - CLI JSON output contract tests

### Task T12

- Description: Introduce project packaging, test layout, and reproducible developer tooling.
- Files affected:
  - `pyproject.toml`
  - `tests/unit/*`
  - `tests/integration/*`
  - `tests/fixtures/*`
  - `README.md`
  - `makefile`
- Dependencies: T01 through T11 can start partial adoption earlier, but full completion should follow them
- Risk: Medium
- Acceptance criteria:
  - Project installs with dev dependencies from a single manifest.
  - Test, lint, and type-check commands are documented and runnable.
  - CI-ready test layout exists for the refactored package.
- Required tests:
  - `pytest`
  - import smoke test
  - CLI smoke tests

### Task T13

- Description: Align container/runtime packaging with the new path model.
- Files affected:
  - `docker/Dockerfile`
  - `docker/docker-compose.yml`
  - `Containerfile`
  - `compose.yml`
  - `README.md`
- Dependencies: T02, T11, T12
- Risk: Medium
- Acceptance criteria:
  - Container starts from repo root without `cd src`.
  - Runtime path semantics match local execution.
  - Compose-level service names and environment defaults are internally consistent.
- Required tests:
  - Container smoke test for CLI help
  - Integration test for config resolution inside container

## Recommended implementation order

1. T01 package skeleton
2. T02 configuration and path stabilization
3. T03 dataset decomposition
4. T04 URL normalization and feature schema
5. T05 leakage fix in dataset build pipeline
6. T06 model/artifact decomposition
7. T07 training and prediction pipelines
8. T08 MLflow lifecycle split
9. T09 artifact identity and OBO compatibility
10. T10 DB and queue adapters
11. T11 CLI wrappers
12. T12 packaging and test tooling
13. T13 container alignment

## Key design decisions

- Keep binary label-space behavior as the default target until the business decision on multiclass support is explicit.
- Treat MLflow as the experiment system of record, but keep local manifest-based artifacts as the operational inference fallback.
- Move to manifest-based model packages incrementally, with legacy artifact compatibility until older models are retired.
- Keep OBO training as a first-class artifact mode only if prediction consumers need it; otherwise leave it as an offline comparison flow with explicit boundaries.

# Session Handoff

## Target architecture
- Introduce `src/semd_ml/` as the real package and keep current `src/` modules as compatibility wrappers during migration.
- Split current mixed modules into configuration, data, features, models, mlops, pipelines, services, interfaces, and infra layers.
- Standardize on manifest-based model packages so training and inference share feature schema, preprocessing, and model identity.
- Make all path resolution repo-root-relative or config-driven so local, CLI, worker, and container runtime semantics match.

## Files to create
- `docs/architecture.md`
- `docs/refactoring-plan.md`
- `pyproject.toml`
- `configs/app.yaml`
- `configs/features.yaml`
- `configs/data_dict.yaml`
- `src/semd_ml/`
- `tests/unit/`
- `tests/integration/`
- `tests/fixtures/`

## Files to modify
- `src/main.py`
- `src/__init__.py`
- `src/core/config.py`
- `src/data/dataset_pipeline.py`
- `src/features/feature_extractor.py`
- `src/ml/ml_pipeline.py`
- `src/ml/training_service.py`
- `src/ml/prediction_service.py`
- `src/tracking/mlflow_tracker.py`
- `src/infra/database.py`
- `src/infra/redis_client.py`
- `src/queues/queue_manager.py`
- `src/workers/queue_worker.py`
- `src/cli/main.py`
- `src/cli/commands/*.py`
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `README.md`
- `makefile`

## Interfaces to preserve
- Current CLI command names and primary flags
- Current training and prediction `job_data` payload shapes
- Prediction response JSON shape
- Ability to reference models by MLflow run ID during transition
- Local artifact loading for existing `models/*.pkl` outputs

## Implementation order
1. Add the package skeleton and compatibility wrappers.
2. Stabilize config and path resolution.
3. Split dataset logic and fix train/test leakage.
4. Introduce URL normalization, feature schema, and manifest-based artifacts.
5. Decompose training, evaluation, inference, and MLflow lifecycle.
6. Repair model identity, OBO loading, DB/queue adapters, CLI wrappers, and container packaging.

## Main risks
- Leakage fix will change reported model metrics and may expose weaker real-world performance.
- Moving off cwd-relative paths can uncover undocumented assumptions in scripts, Docker, and queue workers.
- Artifact migration must support both legacy and manifest-based models until old runs are retired.
- Backend integrations may depend on undocumented model ID semantics beyond what the current code makes explicit.

## Recommended next section
Section 3 — Dataset and Feature Pipeline
