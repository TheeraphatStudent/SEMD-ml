from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from data import dataset_pipeline


class TrainingPipeline:
    """The deterministic dataset-prep -> train-and-select-best-model sequence.

    No MLflow run lifecycle, no report/plot file generation -- those are
    TrainingService's job (see ml/training_service.py), since they're about
    the training *job* (provenance, CLI/queue contract), not about how a
    dataset becomes a trained model.
    """

    def prepare_dataset(
        self,
        dataset_files: List[str],
        balance_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        return dataset_pipeline.prepare_dataset(
            dataset_files=dataset_files,
            apply_balancing=True,
            manual_balance_method=balance_method,
        )

    def train(
        self,
        dataset_result: Dict[str, Any],
        algorithms: Iterable[str],
        run_id: Optional[str] = None,
        git_commit_sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Deferred import: same pre-existing tracking/ml circular-import hazard
        # noted in pipelines/prediction_pipeline.py -- ml.ml_pipeline pulls in
        # the full `ml` package, which imports ml.training_service, which
        # imports this module. Importing at call time instead of module load
        # time avoids adding a new trigger for that cycle.
        from ml.ml_pipeline import ml_pipeline

        return ml_pipeline.train_models(
            dataset_result=dataset_result,
            algorithms=algorithms,
            run_id=run_id,
            git_commit_sha=git_commit_sha,
        )


training_pipeline = TrainingPipeline()
