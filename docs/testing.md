# Testing

## Setup

```bash
uv sync --extra tracking --extra xgboost --group dev
```

The `dev` dependency group provides `ruff`, `mypy`, and `pytest`. The `tracking` extra installs `mlflow`, which the
integration suite requires — without it, `tests/integration/test_mlflow_tracking.py` skips instead of running.

## Running the suite

```bash
uv run --extra tracking --extra xgboost pytest -v
```

Always pass `--extra tracking --extra xgboost` (or run after a `uv sync` that included them). A bare `uv run pytest`
can silently re-sync to an environment without `mlflow`, and the MLflow integration tests will report as **skipped**,
not failed — a skip is not a pass. Check the pytest summary line for a skip count of zero.

Current baseline: **67 passed, 0 skipped, 0 failed**.

## Test layout

```
tests/
├── unit/
│   ├── test_config.py            # MLflow/promotion settings parsing
│   ├── test_dataset_pipeline.py  # validation, cleanup, splitting, hashing
│   ├── test_features.py          # URL normalization, feature extraction, schema
│   ├── test_model_registry.py    # register/promote/rollback, gates, champion loader
│   ├── test_prediction_service.py# predict/batch_predict monitoring-event wiring, best-effort recording
│   ├── test_monitoring_store.py  # prediction-event SQLite store (record/feedback/review/query)
│   ├── test_monitoring_metrics.py# monitoring aggregate metrics (counts, latency percentiles, FPR/FNR)
│   ├── test_retraining.py        # approved-feedback -> labeled CSV dataset construction
│   ├── test_training_pipeline.py # model factory, training, metrics, save/load
│   └── test_url_regression.py    # canonical URL-shape regression table
└── integration/
    └── test_mlflow_tracking.py   # real MLflow run creation, params/metrics/artifacts
```

### Coverage map

| Area | Test(s) |
|---|---|
| URL normalization | `test_features.py::URLNormalizationTests`, `test_url_regression.py` |
| IP detection | `test_features.py::test_ip_detection`, `test_url_regression.py::test_ip_based_url` |
| Punycode | `test_features.py::test_punycode_detection`, `test_url_regression.py::test_punycode_url` |
| Port handling | `test_features.py::test_port_handling`, `test_url_regression.py::test_url_with_explicit_port` |
| Suspicious extensions | `test_features.py::test_suspicious_extension_detection`, `test_url_regression.py::test_suspicious_file_extension_url` |
| Feature extraction / feature order | `test_features.py::FeatureExtractionTests`, `test_dataset_pipeline.py::test_pipeline_feature_order_is_deterministic` |
| Invalid URL | `test_features.py::test_invalid_url_extracts_schema_aligned_defaults`, `test_dataset_pipeline.py::test_invalid_url`, `test_url_regression.py::test_malformed_url`/`test_empty_url` |
| Dataset cleanup | `test_dataset_pipeline.py::test_dataset_cleanup_drops_invalid_duplicate_and_conflicting_rows` |
| Duplicate detection | `test_dataset_pipeline.py::test_duplicate_detection` |
| Conflicting labels | `test_dataset_pipeline.py::test_conflicting_labels` |
| Dataset splitting | `test_dataset_pipeline.py::test_split_avoids_registered_domain_leakage` |
| Model factory | `test_training_pipeline.py::ModelFactoryTests` |
| Metrics | `test_training_pipeline.py::test_metrics_are_complete`, `test_cross_validation_metrics_are_reported` |
| Promotion gates | `test_model_registry.py::test_promotion_gates_reject_bad_candidate` |
| MLflow settings | `test_config.py::MLflowSettingsTests` |
| Fixture dataset training | `test_training_pipeline.py::test_training_with_fixture_data_runs_two_algorithms` |
| MLflow run creation / logging | `test_mlflow_tracking.py::test_mlflow_training_run_logs_required_metadata_and_artifacts` |
| Model registration | `test_model_registry.py::test_register_promote_and_rollback_workflow` |
| Champion loading | `test_model_registry.py::test_champion_loader_caches_model_and_supports_local_fallback` |
| Prediction | `test_training_pipeline.py::test_prediction_output_shape` |
| Schema mismatch | `test_model_registry.py::test_schema_mismatch_is_rejected`, `test_training_pipeline.py::test_feature_schema_incompatibility_raises` |
| Promotion | `test_model_registry.py::test_register_promote_and_rollback_workflow` |
| Rollback | `test_model_registry.py::test_register_promote_and_rollback_workflow` |
| Local fallback | `test_model_registry.py::test_champion_loader_caches_model_and_supports_local_fallback`, `test_loader_raises_when_registry_is_unavailable_and_fallback_is_disabled` |
| Prediction-event recording | `test_monitoring_store.py::test_record_event_persists_all_required_fields`, `test_url_hash_is_stable_and_case_insensitive`, `test_prediction_service.py` (predict/batch_predict wiring, store-failure isolation) |
| User feedback / admin review | `test_monitoring_store.py::test_set_user_feedback_updates_row_and_rejects_unknown_values`, `test_set_admin_review_updates_label_and_timestamp` |
| Monitoring metrics (counts, latency, FPR/FNR) | `test_monitoring_metrics.py` (all cases) |
| Approved-feedback dataset construction | `test_retraining.py` (all cases) |

### Regression URL table (`test_url_regression.py`)

| Case | URL shape |
|---|---|
| Normal HTTPS URL | `https://www.example.com/home` |
| HTTP URL | `http://www.example.com/home` |
| IP-based URL | `http://192.168.1.10/admin` |
| Punycode URL | `https://xn--80ak6aa92e.com/login` |
| Suspicious file extension | `https://example.com/invoice.exe` |
| Long phishing-style URL | 150+ character path with credential-harvesting tokens |
| URL with explicit port | `http://example.com:8080/path` |
| Malformed URL | `http://[not-valid` |
| Empty URL | `""` |

Every case asserts the extractor never raises and always returns the full schema-aligned feature vector, in addition
to case-specific flags (`ip_address_flag`, `punycode_domain_flag`, `port_in_url_flag`, etc.).

## Quality commands

Run from the repo root:

```bash
uv sync --extra tracking --extra xgboost --group dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run --extra tracking --extra xgboost pytest
```

| Command | Status |
|---|---|
| `uv sync` | passes |
| `uv run ruff check .` | passes (0 errors) |
| `uv run ruff format --check .` | fails on files untouched by this work — see "Known limitations" |
| `uv run mypy src` | fails with a known, pre-existing baseline — see "Known limitations" |
| `uv run pytest` | passes (67 passed, 0 skipped) |

`ruff` and `mypy` were not previously installed or configured in this project; `[dependency-groups].dev` and
`[tool.ruff]` / `[tool.mypy]` in `pyproject.toml` were added so these commands are runnable at all.

## Known limitations

- **`ruff format --check .`**: a large number of pre-existing files (not touched by this work) are not yet
  `ruff format`-clean. Files edited as part of this effort are formatted and pass individually. Reformatting the
  entire tree is a separate, dedicated cleanup — bundling it here would produce a large, unrelated diff.
- **`mypy src`**: reports 119 pre-existing-pattern type errors across 18 files (was 109/16 before the Section 9
  monitoring/retraining work), dominated by two patterns that are cosmetic rather than functional bugs:
  - `src/core/config.py` (60 errors, +4): `pydantic.Field(..., env="X")` — a pydantic v1-style kwarg pydantic-settings
    v2 still accepts at runtime (as a deprecated extra) but mypy's v2 stubs no longer recognize. Fixing this means
    touching every setting in the app's central config module, which is out of scope for a testing/docs pass. The
    two new `monitoring_db_path`/`monitoring_dataset_dir` settings reuse this same existing `env=` convention, so
    they reproduce the existing error rather than introducing a new one.
  - `src/tracking/model_registry.py` / `mlflow_tracker.py` (32 errors, unchanged): `self.client: Any | None`
    attribute access (`union-attr`) from the `try: import mlflow / except: mlflow = None` optional-dependency
    pattern — mypy can't narrow the type across the runtime `_require_registry()` guard.
  - `cli/common.py::emit_result`'s `output: str = None` parameter is a pre-existing invalid default for its
    declared type, so every call site passing `getattr(args, "output", None)` trips `arg-type`. Section 9 added
    three CLI commands (`monitor`, `feedback`, `review`: 1 each; `retrain`: 2; `gate-check`: 1) that follow the
    exact same `emit_result` convention every existing command uses, surfacing 6 more instances of this one
    pre-existing bug. Fixing the root cause means changing `emit_result`'s signature once, touching every CLI
    command file — judged out of scope for this pass, same as the two patterns above.
  - No behavior changes were made to silence these; they are documented here as a baseline for a future
    typing-focused pass rather than suppressed with blanket `# type: ignore`.
