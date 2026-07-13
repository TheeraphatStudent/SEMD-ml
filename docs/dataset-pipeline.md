# Dataset Pipeline

## Scope

The dataset pipeline now uses one deterministic flow for training:

1. Load raw CSV/XLSX sources.
2. Standardize source columns to `url` and `raw_label`.
3. Validate raw records and emit dataset-quality metrics.
4. Normalize URLs before deduplication and conflict detection.
5. Drop invalid URLs, invalid labels, missing labels, and conflicting-label URLs.
6. Extract features with the canonical feature schema.
7. Split before balancing.
8. Balance only the training partition.
9. Persist dataset metadata and feature outputs.

## Validation checks

Implemented checks:

- Missing URLs
- Empty URLs
- Invalid URLs
- Duplicate normalized URLs
- Conflicting labels on the same normalized URL
- Missing labels
- Invalid labels
- Empty `type` values
- Class imbalance
- Unique registered-domain count
- Dataset source metadata

Validation entrypoint:

- `src/data/dataset_pipeline.py`
- `src/semd_ml/data/validators.py`

Behavior:

- Missing, empty, invalid, and conflicting-label records are validation errors.
- Duplicates and empty `type` values are reported as warnings.
- Validation stores row-count summaries and example invalid URLs.

## URL normalization policy

Implemented in `src/semd_ml/features/url_normalizer.py`.

Rules:

- Leading and trailing whitespace is stripped.
- Missing schemes are normalized to `http://`.
- Scheme and hostname casing are lowercased.
- URL fragments are removed.
- Unicode hostnames are converted to Punycode with `idna`.
- Default ports are removed from the normalized URL.
- Non-default explicit ports are preserved.
- Invalid hostnames, schemes, and ports return a structured invalid result.

Normalization decisions:

- Query ordering is not normalized.
- Trailing slashes are not normalized.

Rationale:

- Query ordering can be semantically meaningful.
- Trailing slash changes can alter origin routing behavior.

## Split strategy

Implemented in `src/semd_ml/data/splitters.py`.

Primary strategy:

- `StratifiedGroupKFold` holdout using `registered_domain` as the group key.

Fallback:

- Stratified random split when grouped stratification is not feasible.

Leakage rule:

- The same registered domain should not appear in both train and test when the grouped split is possible.

## Balancing policy

- Imbalance is detected on the training partition only.
- Test data is never balanced or synthetically altered.
- Supported methods remain:
  - `none`
  - `smote`
  - `oversampling`
  - `undersampling`

## Dataset metadata

Implemented in `src/semd_ml/data/versioning.py`.

Stored fields:

- `dataset_version`
- `dataset_hash`
- `total_records`
- `valid_records`
- `invalid_records`
- `duplicate_count`
- `conflicting_label_count`
- `benign_count`
- `malicious_count`
- `unique_domains`
- `created_timestamp`
- `source_references`

Hashing policy:

- Stable SHA-256 over sorted cleaned records and sorted source references.
- Input row order does not affect the hash.

## Generated quality report

Generated from `src/dataset/raw/malicious_url_train2.csv` using the first 1000 rows.

Summary:

- Total records: `1000`
- Valid records after cleaning: `987`
- Invalid URLs: `13`
- Duplicate normalized URLs: `0`
- Conflicting labels: `0`
- Benign: `751`
- Malicious: `236`
- Unique registered domains: `770`
- Class imbalance ratio: `3.1822`
- Imbalance severity: `mild`
- Dataset hash: `acafc6d42709098035af1930ab8390d34918b337fba1d7888c4215a946ce6b9f`

Artifacts:

- `reports/dataset_quality_report_malicious_url_train2_first1000.json`
- `docs/dataset-quality-report.md`
