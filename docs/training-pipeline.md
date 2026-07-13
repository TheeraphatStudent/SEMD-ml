# Training Pipeline

The training pipeline now uses canonical model identifiers:

- `svm`
- `random_forest`
- `gradient_boosting`
- `xgboost` when `XGBClassifier` is installed

## Flow

1. Load and standardize source datasets.
2. Validate and clean records.
3. Extract features and align them to the feature schema.
4. Split data into train, validation, and test sets.
5. Apply balancing to the training split only.
6. Train each algorithm with an `imblearn.pipeline.Pipeline` containing:
   - `StandardScaler`
   - optional sampler
   - estimator
7. Run stratified cross-validation, using stratified group folds when domain groups are available.
8. Select the best algorithm by validation malicious-class F1.
9. Save one inference artifact containing the preprocessing pipeline, estimator, feature schema, and metadata.

## Reproducibility

- Random seeds come from `settings.random_state`.
- All splitters use deterministic random states.
- Cross-validation uses the same preprocessing pipeline as normal training.
- Scaling and balancing happen inside the pipeline to prevent leakage.

## Artifact

Each artifact is a single `.joblib` file containing:

- fitted preprocessing and estimator pipeline
- feature schema and expectations
- label encoder classes
- metadata for dataset version, schema version, git SHA, timestamp, metrics, and configuration
