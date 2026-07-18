# Refactoring Plan

Date: 2026-07-14
Project: `semd-ml`

## Update 2026-07-18: T01 reversed

The `src/semd_ml/` package skeleton (T01) has been removed. Target layout is
now flat under `src/*` (`data/`, `features/`, `pipelines/`, `bootstrap.py`
moved into `cli/bootstrap.py`), matching the pre-existing module layout
(`core/`, `ml/`, `tracking/`, `infra/`, `workers/`, `cli/`) instead of a
separate nested package. Every `semd_ml.X.Y` reference below should be read
as `X.Y` against `src/` directly. T02-T13 file lists that mention
`src/semd_ml/...` paths are stale in that respect; the module split they
describe still applies, just without the `semd_ml/` prefix.

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
- `PredictionService.execute_prediction()` response shape (locked by
  `tests/unit/test_queue_worker.py::test_successful_job_result_shape_is_locked`;
  this is the queue-worker's wrapped result on `ml_result_queue`, not
  `execute_prediction()`'s own return value — see that test for the raw shape):
  - `status`
  - `url`
  - `prediction` (nested dict: `prediction.prediction` (class string), `prediction.is_malicious`,
    `prediction.confidence`, `prediction.model_version`, `prediction.model_alias`, ... — no
    `prediction.probabilities` key exists today)
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
  - `src/ml/model_factory.py` (renamed from the plan's `src/semd_ml/models/factory.py` — pre-existed already, flat layout, see 2026-07-18 T01-reversed update above)
  - `src/ml/training.py` (was `src/semd_ml/models/training.py`)
  - `src/ml/evaluation.py` (was `src/semd_ml/models/evaluation.py`)
  - `src/ml/inference.py` (was `src/semd_ml/models/inference.py`)
  - `src/ml/artifacts.py` (was `src/semd_ml/models/artifacts.py`)
  - ~~`src/semd_ml/models/package_manifest.py`~~ — not built, see status
- Dependencies: T05
- Risk: High
- Status (2026-07-18): **Split into (a) decomposition and (b) manifest packaging on
  advisor review, because they have opposite risk profiles and bundling them would make
  a predict-path regression impossible to attribute. (a) is done and verified live. (b)
  is not started — this is a deliberate checkpoint, not a partial/abandoned attempt.**
  - **(a) done:** `MLPipeline` (previously a 431-line class doing everything) now
    delegates to `TrainingPipelineBuilder` (`ml/training.py` — pipeline construction +
    cross-validation, one scaler step shared by both), `ml/evaluation.py` (pure metric
    functions), `ArtifactStore` (`ml/artifacts.py` — `.joblib` save/load/path-resolution
    I/O), and `ml/inference.py` (single-URL prediction). `MLPipeline` itself is now the
    stateful orchestrator (best_model, label_encoder, loaded_artifact, ...) plus the
    `train_models()` loop — every existing method name/signature on `MLPipeline` and the
    `ml_pipeline` singleton is preserved as a thin wrapper, since `model_registry.py`
    (`pipeline_factory=MLPipeline`), `training_service.py`, and `cli/common.py` all call
    into it directly. `.joblib` artifact format is completely unchanged by this pass.
    Verified with the real train → register → promote/predict cycle against live
    mlflow/redis/postgres (not just pytest): trained a fresh run, registered it, predicted
    via both MLflow version and local `run_id` (T09 path) — all correct. Also loaded
    `models/svm_run_20260713194125_8e31ecfe.joblib`, a real artifact saved before this
    decomposition existed, directly through the new `MLPipeline.load_artifact` and got a
    correct prediction — the actual legacy-artifact-loading regression check, using a real
    pre-existing file rather than a fixture (see the "Required tests" note below on why no
    fixture-based version was added).
  - **(b) not started:** no manifest-based package format, no dual legacy/manifest loading
    path. `model_registry.py::_resolve_model_source` hard-requires a `.joblib` artifact
    logged to the MLflow run (`artifact.path.endswith(".joblib")`) — introducing a manifest
    format has to account for that coupling or `register`/`promote`/`predict` breaks. Also
    checked before assuming scope: `../models/` has an older xgboost *triplet* format
    (`xgboost_model_*.pkl` + `xgboost_scaler_*.pkl` + `xgboost_label_encoder_*.pkl`)
    alongside the current unified `.joblib` dict — no code path loads the triplet today
    (`load_artifact` does `joblib.load(...)["pipeline"]`, which the triplet was never
    compatible with), so "legacy artifacts" for (b)'s dual-loading criterion means the
    current `.joblib` format only, not the triplet. That triplet appears to be dead weight
    from before this repo's `.joblib`-dict format existed; not touched here since deleting
    old model files is outside a packaging task's scope.
- Acceptance criteria:
  - Scaling occurs in one place only. — done (`TrainingPipelineBuilder.build_pipeline`, single scaler step, locked by `test_ml_training_builder.py::test_scaling_happens_in_exactly_one_place`)
  - Training emits a manifest-based model package. — not started, part of (b)
  - Inference can load both legacy artifacts and new model packages. — not applicable until (b) exists a second format to be compatible with
- Required tests:
  - Unit tests for algorithm factory output — done: `tests/unit/test_model_factory.py` (new; `model_factory.py` itself pre-existed but had no dedicated test file, only indirect coverage via `test_training_pipeline.py`)
  - Integration test for train-save-load-predict round trip — already existed and still passes: `tests/unit/test_training_pipeline.py` ("roundtrip-run"/"predict-run" cases); also re-verified live against real infra (see status above)
  - Regression test for legacy artifact loading — verified manually against a real pre-decomposition `.joblib` file (see status above), not captured as an automated pytest because doing so would mean committing a binary model fixture to git for a format that hasn't changed; revisit once (b) introduces an actual second format worth diffing against in CI

### Task T07

- Description: Separate training and prediction workflows into dedicated pipelines while preserving service entrypoints.
- Files affected:
  - `src/ml/training_service.py`
  - `src/ml/prediction_service.py`
  - `src/pipelines/training_pipeline.py` (was `src/semd_ml/pipelines/training_pipeline.py`)
  - `src/pipelines/prediction_pipeline.py` (was `src/semd_ml/pipelines/prediction_pipeline.py`)
  - ~~`src/semd_ml/services/training_service.py`~~ / ~~`src/semd_ml/services/prediction_service.py`~~ — not built, see status
- Dependencies: T05, T06
- Risk: Medium
- Status (2026-07-18): **Done**, with one intentional deviation from the file list.
  - `src/pipelines/training_pipeline.py`: new `TrainingPipeline` wraps the deterministic
    `dataset_pipeline.prepare_dataset()` -> `ml_pipeline.train_models()` sequence, with no
    MLflow run lifecycle or report/plot generation. `TrainingService.execute_training()`
    now calls `training_pipeline.prepare_dataset()`/`.train()` at the same two points in
    its flow instead of calling `dataset_pipeline`/`ml_pipeline` directly — the MLflow
    `start_run` call still happens between them (needs `dataset_result` for tags), so the
    two pipeline calls stay separate rather than being collapsed into one `.run()`.
  - `src/pipelines/prediction_pipeline.py`: new `PredictionPipeline` owns model resolution
    (reuse the already-loaded model, reload if a different `model_id` is requested, fall
    back to `"champion"` if nothing is loaded yet) + a single inference call. `monitoring_store`
    event recording and `batch_predict`'s per-URL loop stay in `PredictionService` — those
    are job/service concerns (CLI/queue contract, monitoring), not part of resolving a
    model and running inference. `PredictionService.model_loader`/`.current_model_id` are
    now properties proxying to the pipeline instance (get *and* set) rather than direct
    attributes, because `tests/unit/test_prediction_service.py` patches them directly
    (`service.model_loader = FakeModelLoader(...)`) — properties preserve that external
    contract exactly.
  - **Deviation:** did not create `src/semd_ml/services/*.py` (or a flat `src/services/*.py`
    equivalent). `src/ml/training_service.py`/`prediction_service.py` already are the
    service layer per this repo's actual (flat, T01-reversed) module map documented in
    `/CLAUDE.md` — moving them to a same-purpose file in a new directory just to match the
    plan's original nested-package file list would be a rename with no behavior value,
    same reasoning as T11's dropped `interfaces/cli` move.
  - **Bug found and fixed during this work, not part of the plan:** constructing either
    new pipeline module as the *first* import in a process (e.g. `import
    pipelines.prediction_pipeline` before anything touches the `ml` package) triggered a
    circular import: `pipelines.prediction_pipeline` -> `tracking.model_registry` ->
    `tracking/__init__` -> `mlflow_tracker` -> `ml.ml_pipeline` -> (`ml` package init) ->
    `ml.prediction_service` -> `pipelines.prediction_pipeline` (still mid-import). This is
    an existing fragility in this codebase — `tests/unit/test_queue_worker.py` already has
    a comment documenting the same `tracking/mlflow_tracker.py` <-> `tracking/model_registry.py`
    cycle and works around it with import ordering — but introducing the new `pipelines.*`
    modules gave it a new way to actually trigger. Fixed by deferring the
    `tracking.model_registry`/`ml.ml_pipeline` imports in both new pipeline modules to
    inside the function/method that needs them instead of module top level, so importing
    either module standalone no longer forces the rest of the import graph to resolve
    immediately. Verified with a cold-process import-order stress test
    (`uv run python -c "from pipelines.prediction_pipeline import PredictionPipeline"` etc.,
    each as the sole first import), not just pytest's own collection order. The underlying
    `tracking/mlflow_tracker.py` <-> `tracking/model_registry.py` cycle itself is untouched
    and still exists — fixing that root cause is a `tracking/` module concern, out of scope
    here.
  - Verified live against real infra, not just pytest: full train -> register -> predict
    cycle (both MLflow-version and local-`run_id` resolution), plus a queue-worker
    round-trip (pushed a prediction job to Redis, ran `worker --mode prediction`, confirmed
    the structured `ml_result_queue` result via the champion alias).
- Acceptance criteria:
  - Existing job payloads still work. — done, verified live
  - Services become thin orchestration layers over pipelines. — done for the pieces that were doing real work (dataset-prep+train call; model-resolution+inference); report/plot generation and monitoring-event recording deliberately stayed in the services, see status
  - Prediction result shape remains backward compatible. — done, unchanged (`PredictionService.execute_prediction`/`batch_predict` signatures and return shapes untouched; also covered by T11's `test_cli_output_contract.py` and `test_queue_worker.py::test_successful_job_result_shape_is_locked`)
- Required tests:
  - Integration tests for CLI-equivalent training job payload — already existed (`tests/integration/test_mlflow_tracking.py`, still passing) plus new unit coverage: `tests/unit/test_training_pipeline_delegation.py`
  - Integration tests for single and batch prediction payloads — already existed (`tests/unit/test_prediction_service.py`, still passing) plus new unit coverage: `tests/unit/test_prediction_pipeline.py`

### Task T08

- Description: Refactor MLflow integration into tracking, registry, and promotion responsibilities.
- Files affected:
  - `src/tracking/mlflow_tracker.py` (tracking, already its own file pre-refactor)
  - `src/tracking/registry.py` (was `src/semd_ml/mlops/registry.py` — new `Registry` class, pure CRUD)
  - `src/tracking/promotion.py` (was `src/semd_ml/mlops/promotion.py` — new `Promotion` class, gate/validation policy)
  - `src/tracking/model_registry.py` (kept as a compatibility facade — see status)
  - ~~`src/semd_ml/mlops/lineage.py`~~ — not built, see status
- Dependencies: T06, T07
- Risk: Medium
- Status (2026-07-18): **Done.** `tracking/model_registry.py`'s 502-line `ModelRegistryManager`
  mixed registry CRUD, promotion-gate policy, and champion-loading-with-cache into one
  class/file. Split into:
  - `tracking/registry.py`: `Registry` — register a run as a candidate version, resolve
    alias/version/run_id references, rollback, load a referenced artifact into a pipeline.
    No gate/threshold/champion-comparison logic at all (locked by
    `test_tracking_registry_promotion.py::test_registry_has_no_gate_or_promotion_methods`
    and an AST-based check that `registry.py` has no module-level import of `promotion.py`).
  - `tracking/promotion.py`: `Promotion` — gate evaluation, champion comparison, smoke
    tests, `validate_candidate`/`promote_candidate`. Takes a `Registry` instance; the
    dependency is one-directional (`Promotion` -> `Registry`, never the reverse).
  - `tracking/model_registry.py` kept as a thin facade: `ModelRegistryManager` composes a
    `Registry` + `Promotion` and re-exposes every method/property the original class had
    (`register_candidate`, `validate_candidate`, `promote_candidate`,
    `rollback_to_previous_champion`, `load_reference`, `client`, `available`, ...) —
    every existing caller (`cli/commands/model_registry.py`, `CachedChampionModelLoader`,
    `tests/unit/test_model_registry.py`'s `FakeMlflowClient`-based suite) works completely
    unchanged. `CachedChampionModelLoader` itself stayed in `model_registry.py` rather
    than moving to a separate file: it's tightly duck-typed (accepts *any* object with a
    `.load_reference()` method, not just a real `Registry`/`ModelRegistryManager` — see
    `test_model_registry.py`'s `FailingRegistry` fixture) and directly attribute-tested
    (`loader._cached_pipeline`), so moving it would be a rename with no decomposition
    value, not a real win.
  - No `lineage.py` was built as a separate module — dataset lineage (`dataset_version`,
    `dataset_hash`, `feature_schema_version`) is already captured as MLflow tags at
    `register_candidate()` time and read back during `validate_candidate()`; it didn't
    need its own file, it needed the tags it already had.
  - **Root-caused and fixed the tracking/ml circular-import fragility that T07 had only
    worked around.** T07's status notes flagged a cycle
    (`pipelines.prediction_pipeline` -> `tracking.model_registry` -> `tracking/__init__`
    -> `mlflow_tracker` -> `ml.ml_pipeline` -> `ml` package init -> `ml.prediction_service`
    -> back to `pipelines.prediction_pipeline`, still mid-import) and deferred two imports
    to work around it. Investigating further for T08 found the actual root: `MLflowTracker`
    had a completely dead `register_model()` method (never called anywhere in src/ or
    tests/) whose only purpose was importing `ModelRegistryManager` from
    `tracking.model_registry` — that import existed purely to support unreachable code.
    Deleted `register_model()`/`evaluate_model()` (the latter was also dead, always
    returned `None`) and their import. That fixed one trigger, but `tracking/model_registry.py`
    itself still imported `MLPipeline` from `ml.ml_pipeline` at module top level, which was
    the *other* independent trigger of the same cycle (confirmed by tracing it through with
    the pre-refactor file structure — this was never a T06/T07/T08-introduced bug, it
    predates this session's work). Deferred that import too (function-scope, `TYPE_CHECKING`
    for the type hint). Verified with a cold-process stress test importing each of
    `tracking.model_registry`, `tracking.mlflow_tracker`, `tracking` (package),
    `tracking.registry`, `tracking.promotion`, `pipelines.prediction_pipeline`,
    `pipelines.training_pipeline`, and `ml` as the *sole first import* in a fresh process —
    all 8 pass now, including reversing the exact ordering `tests/unit/test_queue_worker.py`
    used to depend on (its now-unnecessary "import first" workaround comment was removed).
  - Verified live against real infra: train -> register -> promote (including a real gate
    rejection — a candidate's `prediction_latency_ms` narrowly missed the champion-relative
    threshold, 0.626ms vs 0.609ms — correctly evaluated and reported per-metric, not a bug)
    -> `gate-check` dry run -> predict via champion, plus a queue-worker round trip through
    Redis.
- Acceptance criteria:
  - Tracking run logging works independently of model promotion. — done, and now true for real: `mlflow_tracker.py` has zero imports from `tracking.model_registry`/`tracking.promotion`
  - Model registration stores manifest metadata and dataset lineage. — done (dataset_version/dataset_hash/feature_schema_version tags at registration; "manifest" in the packaging sense is still T06 part (b), not started)
  - Alias/stage transitions are policy-driven and optional. — done (`settings.parsed_model_promotion_gates`, `settings.promotion_require_champion_comparison` — champion comparison is explicitly optional and configurable, this predates T08 but is now verified to still work through the split)
- Required tests:
  - Unit tests with mocked MLflow client — already existed (`tests/unit/test_model_registry.py`'s `FakeMlflowClient`, still passing through the facade) plus new direct coverage: `tests/unit/test_tracking_registry_promotion.py`
  - Integration test for run logging and model registration metadata — already existed (`tests/integration/test_mlflow_tracking.py`, still passing) plus this session's live verification against real infra

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
  - ~~`src/semd_ml/infra/database.py`~~ / ~~`src/semd_ml/infra/redis.py`~~ / ~~`src/semd_ml/services/worker_service.py`~~ — dead targets, see status
- Dependencies: T07
- Risk: Medium
- Status (2026-07-18): **Done**, and smaller than the plan implied once the actual code
  was checked — no `infra/`/`services/` restructuring was needed.
  - **Queue status decode bug**: fixed earlier this session (see T09/general session
    notes) — `queue_manager.py` called `job.decode('utf-8')` unconditionally, but
    `infra/redis_client.py` configures `decode_responses=True`, so `lrange` already
    returns `str`; every queued job was silently degraded to an `{"error": ...}` entry
    instead of being parsed. Fixed to check `isinstance(job, bytes)` first. That fix had
    no dedicated test until now — added `tests/unit/test_queue_manager.py` (5 tests:
    str payload, bytes payload for defense, malformed JSON, empty queue, all three
    queue names present), satisfying this task's "Unit tests for queue status parsing"
    requirement.
  - **"Prediction persistence uses a stable model reference instead of parsing a numeric
    suffix from run ID"**: investigated and found **no such parsing exists anywhere in
    the current codebase** (`grep` for numeric-suffix/regex handling of `run_id` in
    `infra/database.py` and `monitoring/store.py` turned up nothing). The live
    prediction-persistence path is `monitoring/store.py`'s `MonitoringStore.record_event`,
    which stores `model_version` as plain `TEXT` — no parsing at all, so there's nothing
    to fix here. This criterion likely described a bug in a pre-T09 version of
    `prediction_service.py` that this session's earlier `docs/refactoring-plan.md`
    updates (T09) already superseded with real MLflow-registry-based model-identity
    resolution, before this T10 pass ever started.
  - **Found, not fixed — flagging rather than silently deleting:** `src/infra/database.py`'s
    `DatabaseClient`/`db_client` (Postgres `service_conf`/`model_registry`/`prediction`
    tables) is **fully disconnected dead code** — `grep` across `src/` and `tests/` found
    it referenced only by `src/__init__.py`'s backward-compat umbrella re-export, never
    called by `training_service.py`, `prediction_service.py`, any `cli/` command, or
    `workers/queue_worker.py`. The live app persists predictions via `monitoring/store.py`
    (SQLite) instead. Whether `infra/database.py` should be deleted, or is intentionally
    kept for some other consumer (e.g. a `semd-backend` integration outside this repo) is
    a real decision, not a packaging-task call — left in place.
  - **Worker behavior**: read `workers/queue_worker.py` in full; already has explicit
    retryable/non-retryable error classification, graceful-shutdown signal handling, and
    structured failure payloads (`build_job_failure_result`, already covered by
    `tests/unit/test_queue_worker.py`). No changes needed. Verified live three times this
    session (T07, T08 checks, and this pass): pushed a prediction job to Redis, ran
    `worker --mode prediction`, confirmed a correctly structured `ml_result_queue` result
    each time.
- Acceptance criteria:
  - Queue status works with decoded Redis responses. — done
  - Prediction persistence uses a stable model reference instead of parsing a numeric suffix from run ID. — not applicable, no such parsing exists in the current codebase (see status)
  - Worker behavior matches current queue contracts. — done, verified live
- Required tests:
  - Unit tests for queue status parsing — done: `tests/unit/test_queue_manager.py`
  - Integration tests with mocked DB and Redis clients — `tests/unit/test_queue_worker.py` already covers mocked-Redis worker behavior; no DB-client test added since `infra/database.py` is unused dead code (see status) and testing dead code adds no value

### Task T11

- Description: Preserve CLI compatibility while moving command implementations into the new package.
- Files affected:
  - `src/cli/main.py`
  - `src/cli/common.py`
  - `src/cli/commands/*.py`
  - ~~`src/semd_ml/interfaces/cli/*`~~ — dead target, `src/semd_ml/` was removed (T01 reversed, see the 2026-07-18 update above). `cli/` is already a flat top-level package; there is no `interfaces/` nesting to move it into.
- Dependencies: T07, T10
- Risk: Low *only once T07/T10 are done* — see status below.
- Status (2026-07-18): **Partially done.** The move-and-de-singleton part of T11 is still blocked on T06/T07 (undone, T06 is High risk) — the CLI command modules import service *singletons* (`from ml import prediction_service`, `from data import dataset_pipeline`, etc.) all the way down, so "CLI no longer relies on import-time singleton state" can't happen until those services stop being singletons. What's done now is the unblocked, dependency-free slice: the required tests below, which pin the current CLI contract so a later T06/T07/T11 pass has a regression net instead of guessing what "still works" means.
- Acceptance criteria:
  - All existing command names and main flags still work.
  - Output JSON contracts remain unchanged unless explicitly versioned.
  - CLI no longer relies on import-time singleton state. — **not done, blocked on T06/T07**
- Required tests:
  - CLI integration tests for `train --help`, `predict --help`, `evaluate --help` — done, extended to all 19 registered subcommands: `tests/integration/test_cli_bootstrap.py`
  - CLI JSON output contract tests — done: `tests/integration/test_cli_output_contract.py` (locks `main.py predict`/`predict-test` JSON shape) and `tests/unit/test_queue_worker.py::test_successful_job_result_shape_is_locked` (locks the backend-facing `ml_result_queue` payload shape, which is the shape actually named in this doc's "Interfaces to preserve" section above — that section's `prediction.class`/`prediction.is_malicious` nesting is stale relative to the real response and should be corrected there, not treated as current truth)

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
- Status (2026-07-18): **Dev-tooling slice done.** `make venv` now runs `uv sync --extra
  tracking --extra xgboost --group dev` (was `uv pip install -r requirements.txt`, which
  is how mlflow ended up missing during service testing this session — `predict`/`register`/
  `promote` hard-require it but it's an optional extra). README Setup section corrected to
  match. `docs/testing.md` + README's "Testing & Quality" section already documented
  lint/type-check/test commands correctly; only the dev-install path was stale.
  Skipped `tests/fixtures/*` — the one smoke fixture already lives at
  `src/dataset/raw/t093_smoke_fixture.csv` and works fine there; an empty directory
  wouldn't add anything.
  **Not fully done:** "single manifest" is still two files (`pyproject.toml` for dev,
  `requirements.txt` for the Docker image) — `requirements.txt` retirement is deliberately
  left for T13, atomically with the Dockerfile's `uv sync` switch, so the container build
  is never broken by a T12-only change. Also note `CLAUDE.md`'s quickstart still says plain
  `uv sync` (no `--extra tracking`) — same missing-mlflow trap for anyone following it
  literally; not touched here since it's a docs file outside this plan's file list.
- Acceptance criteria:
  - Project installs with dev dependencies from a single manifest. — partial, see status
  - Test, lint, and type-check commands are documented and runnable. — done
  - CI-ready test layout exists for the refactored package. — done (`tests/unit`, `tests/integration`)
- Required tests:
  - `pytest` — done, 115 passing
  - import smoke test — done (`verify_imports.py`, `test_cli_bootstrap.py`)
  - CLI smoke tests — done (`test_cli_bootstrap.py`, extended under T11)

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
- Status (2026-07-18): **Done**, with one deliberate carve-out. `docker/Dockerfile` now
  builds via `uv sync --frozen --no-dev --extra tracking --extra xgboost` against
  `pyproject.toml`/`uv.lock` instead of `pip install -r requirements.txt`, runs from
  `WORKDIR /app` (repo root), and starts via the installed `semd-ml` console script
  instead of `python main.py` — no `cd src` anywhere in the image. Verified with a real
  `podman build` + `podman run` against the live `semd-shared-network`: `pwd` is `/app`,
  `semd-ml --help` and `semd-ml queue-status` both work unmodified, `core.settings`
  paths resolve correctly from the container (`PROJECT_ROOT` is `__file__`-derived, not
  cwd-derived, so this was safe). `docker/docker-compose.yml` already only defines
  `mlflow` (ml-service was dropped from it earlier this session per explicit request) —
  nothing in this repo's compose currently references `Dockerfile` at all; it's a
  standalone buildable artifact now, not wired into the local compose stack. No
  `Containerfile`/`compose.yml` exist in this repo (those filenames are unused here).
  **Carve-out:** `requirements.txt` was *not* deleted, contrary to the "single manifest"
  framing in T12. `src/ml/training_service.py:315-317` reads it live at training time and
  logs its contents as the MLflow `dependency_file` provenance artifact per run — deleting
  it breaks training, not just Docker. It no longer drives any install path (dev or
  container), but it must still be kept manually in sync with `pyproject.toml`/`uv.lock`
  until `training_service.py` is pointed at `uv.lock` instead (not attempted here — that's
  a training-service change, out of scope for a packaging task).
- Acceptance criteria:
  - Container starts from repo root without `cd src`. — done
  - Runtime path semantics match local execution. — done
  - Compose-level service names and environment defaults are internally consistent. — done (nothing left referencing the removed `ml-service` block)
- Required tests:
  - Container smoke test for CLI help — done, verified via `podman run`
  - Integration test for config resolution inside container — done, verified via `podman run` (`core.settings` paths), not yet captured as an automated pytest (would need a container-capable CI runner; today's verification was manual)

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
