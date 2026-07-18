---
type: "explain"
date: "2026-07-17T02:35:56.327955+00:00"
question: "Explain models"
contributor: "graphify"
source_nodes: ["src_ml_ml_pipeline_mlpipeline_train_models"]
---

# Q: Explain models

## Answer

Matched .train_models() (MLPipeline method, src/ml/ml_pipeline.py:102, community 3 ML Pipeline Core, degree 15). It orchestrates training: builds feature schema, builds training pipeline, cross-validates, evaluates each model, then handles artifact bookkeeping (build_artifact_metadata, generate_run_id, artifact_path, save/load_artifact). 14 EXTRACTED edges, 1 INFERRED (call to build_feature_schema).

## Source Nodes

- src_ml_ml_pipeline_mlpipeline_train_models