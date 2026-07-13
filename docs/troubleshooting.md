# Troubleshooting

## Setup

**`uv sync` doesn't install `ruff`/`mypy`/`pytest`**
Those live in the `dev` dependency group, and `mlflow`/`xgboost` are optional extras — none are installed by a bare
`uv sync`. Use:
```bash
uv sync --extra tracking --extra xgboost --group dev
```

**`ruff: command not found` / `mypy: command not found`**
Same cause as above — run the `uv sync` command with `--group dev`, then invoke via `uv run ruff ...` /
`uv run mypy ...` (not the bare binary; `uv run` resolves it inside the project's `.venv`).

**`mypy src` fails with "Source file found twice under different module names"**
This happens if `mypy_path` in `pyproject.toml` also points at `src` while you invoke `mypy src` — mypy then resolves
`src/data/dataset_pipeline.py` both as `src.data.dataset_pipeline` and `data.dataset_pipeline`. The project's
`[tool.mypy]` config uses `explicit_package_bases = true` instead of `mypy_path`, which avoids this; if you see the
error again after editing that config, remove any `mypy_path` override you added.

## Tests

**`tests/integration/test_mlflow_tracking.py` reports as skipped, not passed**
The class is decorated `@unittest.skipIf(mlflow is None, ...)`. If `mlflow` isn't importable in the environment
pytest actually runs in, every test in that file silently skips instead of failing — easy to miss in a long test
log. Run with the extras explicit and check the summary line for `0 skipped`:
```bash
uv run --extra tracking --extra xgboost pytest -v
```

**A plain `uv run pytest` behaves differently between two shells / re-syncs away `mlflow`**
`uv run` re-syncs the environment to match `pyproject.toml`/`uv.lock` before running. If a previous command synced
without `--extra tracking`, the next bare `uv run pytest` can silently drop `mlflow` again. Always pass the extras
explicitly for commands that need them, rather than relying on a previous sync's state persisting.

**Training tests are slow or flaky**
`tests/unit/test_training_pipeline.py` and `test_model_registry.py` train real (tiny fixture) models with
`RandomizedSearchCV`, so they take real CPU time — this is expected, not a hang. If a test fails intermittently on
a metric threshold, check whether `core.settings.random_state` was mutated by another test that didn't restore it in
`tearDown`.

## MLflow / Docker

**MLflow permission errors (`PermissionError` writing to `mlflow_data/` or `models/`)**
```bash
make mlflow-permissions
```
Creates the required directories and `chown`s them to the configured `ML_USER`/`ML_GROUP` (see `makefile`). Requires
`sudo`.

**`make start` fails because the network doesn't exist**
All services expect an external bridge network:
```bash
podman network create semd-shared-network
```
(or the `docker` equivalent if you're using Docker instead of Podman). This is created once, outside any compose
file, and shared across the ML service, backend, frontend, and MLflow containers.

**`ml-service` container never starts / stays waiting**
It has `depends_on: mlflow: condition: service_healthy`. Check the MLflow container's healthcheck directly:
```bash
make logs LOG_SERVICE=mlflow
curl -f http://localhost:5000
```
If MLflow itself won't start, check `MLFLOW_ARTIFACT_ROOT` points at a writable, existing path and that
`mlflow_data/mlflow.db` isn't locked by a stale process.

## Model registry / promotion

**`ModelRegistryError: MLflow model registry is unavailable`**
`ModelRegistryManager` couldn't construct an `MlflowClient` — usually `MLFLOW_TRACKING_URI` is unreachable or
`mlflow` isn't installed in the current environment (`uv sync --extra tracking`). If serving must continue without
registry access, see the local-fallback settings in `docs/model-serving.md` — fallback is off by default and must be
explicitly and fully configured (all four `MLFLOW_LOCAL_FALLBACK_*` vars).

**`ModelValidationError: Candidate feature schema mismatch`**
The registered run's `feature_schema.json` `schema_version` doesn't match the currently running code's schema
version (`semd_ml.features.schema.build_feature_schema`). This means the feature extractor changed between when the
run was trained and now — retrain against the current code before registering, don't force past the check.

**`promote` raises `ModelValidationError: Candidate validation failed`**
One or more of: promotion gates (`MODEL_PROMOTION_GATES`), champion comparison, or smoke tests failed. The full
per-gate breakdown is in the `validate_candidate` response (`gate_results`, `champion_comparison`, `smoke_tests`) —
inspect that before assuming the candidate is simply "bad"; a smoke-test failure can also mean the configured
`PROMOTION_SMOKE_TEST_URLS` produce an unexpected class rather than the model being wrong.

**`rollback` raises `ModelRegistryError: Rollback aborted: no previous champion alias is configured`**
There's nothing to roll back to — `previous-champion` is only set by a prior successful `promote` call. Rollback
intentionally never invents a fallback version (see `docs/rollback.md`).

## Feature / dataset

**`data-migrate` reports `0` extracted files**
Check `src/dataset/store/` actually has archives, and that `dataset_path`/`extraction_path` in `.env` point where
you expect — both default to paths under `src/dataset/`, relative to the `src/` working directory the CLI expects
to be run from.

**Training raises `ValueError: Dataset cleaning removed all rows`**
`DatasetValidator.clean()` dropped every row — check the validation report for `invalid_url_count`,
`missing_label_count`, or `conflicting_label_count` matching your total row count, which usually means a
`data_dict.yaml` column mapping doesn't match the new source file's actual column names.
