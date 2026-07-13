# Model Registry

`semd-ml` now treats MLflow Registry as the source of truth for served models.

## Registered model

- Name: `semd-malicious-url-detector`
- Primary aliases:
  - `candidate`
  - `champion`
  - `previous-champion`

Aliases are preferred over deprecated stage-based promotion.

## Registration

Register a completed MLflow run as a candidate:

```bash
uv run semd-ml register --run-id <run-id>
```

Registration requires a logged `.joblib` artifact under the run's `artifacts/` directory plus:

- `feature_schema.json`
- `dataset_metadata.json`
- `sample_predictions.json`

The command creates an MLflow model version and assigns the `candidate` alias.

## Validation gates

Default gates are configurable through `MODEL_PROMOTION_GATES`:

```json
{
  "malicious_recall": { "operator": ">=", "threshold": 0.95 },
  "malicious_f1": { "operator": ">=", "threshold": 0.93 },
  "false_negative_rate": { "operator": "<=", "threshold": 0.05 },
  "prediction_latency_ms": { "operator": "<=", "threshold": 200.0 }
}
```

Validation checks:

- feature schema compatibility
- dataset metadata presence
- required metrics against gates
- candidate versus current champion on the same gate metrics
- prediction smoke tests

Promotion is rejected when any gate fails. Accuracy alone is never used as the promotion decision.

## Promotion

Promote the aliased candidate:

```bash
uv run semd-ml promote
```

Promote an explicit version:

```bash
uv run semd-ml promote --model-version <version>
```

Workflow:

1. Validate candidate artifacts and metrics.
2. Preserve the existing `champion` as `previous-champion`.
3. Assign the approved version to `champion`.
4. Record promotion metadata as model version tags.
