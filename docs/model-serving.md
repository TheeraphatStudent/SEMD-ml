# Model Serving

Inference loads the `champion` alias from MLflow Registry and keeps the model cached in memory.

## Loading behavior

- Default source: `models:/semd-malicious-url-detector@champion`
- Cache scope: process memory
- Reload trigger:
  - first prediction in a process
  - explicit non-champion selector
  - manual cache clear in code

The service does not load a model on every request.

## Validation

Before a model is served:

- the artifact is downloaded from MLflow Registry
- the packaged feature schema is checked against the runtime schema
- the response includes model identity fields

If MLflow Registry is unavailable, serving fails unless local fallback is explicitly enabled.

## Local fallback

Local fallback is disabled by default.

Enable it only with all of the following configured:

- `MLFLOW_LOCAL_FALLBACK_ENABLED=true`
- `MLFLOW_LOCAL_FALLBACK_MODEL_PATH`
- `MLFLOW_LOCAL_FALLBACK_MODEL_VERSION`
- `MLFLOW_LOCAL_FALLBACK_MODEL_NAME`

This prevents silently serving an unknown version.

## Prediction contract

Single prediction responses now include:

```json
{
  "url": "https://example.com",
  "prediction": "benign",
  "is_malicious": false,
  "confidence": 0.94,
  "model_name": "semd-malicious-url-detector",
  "model_version": "12",
  "model_alias": "champion",
  "feature_schema_version": "2.1.0",
  "prediction_time_ms": 14.2
}
```

The backend queue adapter preserves its existing nested `prediction` envelope while forwarding these fields.
