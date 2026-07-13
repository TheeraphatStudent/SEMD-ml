# Rollback

Rollback promotes `previous-champion` back to `champion`.

## Command

```bash
uv run semd-ml rollback
```

## Safeguards

- rollback aborts when no `previous-champion` alias exists
- rollback does not invent a fallback version
- rollback records model version tags for auditability

## Alias behavior

When rollback succeeds:

1. `previous-champion` becomes `champion`
2. the current `champion` is reassigned to `previous-champion`

This makes rollback reversible and preserves a clear promotion chain.

## Recommended checks

- verify the restored version in MLflow Registry
- run a post-rollback smoke prediction
- confirm backend responses report the restored `model_version`
