"""Regression coverage for T-093 (MLflow artifact persistence).

Requires a live MLflow tracking server reachable at MLFLOW_TRACKING_URI
(`podman compose -f docker/docker-compose.yml up -d mlflow`). Skips itself
entirely if that server is unreachable.

The bug this guards against: MLflowTracker used to pass an explicit,
process-cwd-relative `artifact_location` when creating an experiment
(mlflow_tracker.py's old `_normalize_artifact_root`). That produced a plain
filesystem path baked into the experiment forever. Because the tracking
server was started with --serve-artifacts, the *intent* was that all artifact
I/O goes through the server's HTTP API -- but a plain (non "mlflow-artifacts:")
path bypasses that proxy, so each client wrote straight to its own local,
unmounted disk. Artifacts were unreachable from any other process and lost on
container recreation. See docs/section-10-infrastructure-validation.md for the
full empirical trace.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency guard
    mlflow = None

from core.config import MLServiceSettings

settings = MLServiceSettings()


def _mlflow_server_reachable() -> bool:
    if mlflow is None:
        return False
    try:
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        client.search_experiments(max_results=1)
        return True
    except Exception:
        return False


@unittest.skipUnless(_mlflow_server_reachable(), "Live MLflow tracking server not reachable")
class MlflowArtifactPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.experiment_name = f"t093-regression-{uuid.uuid4().hex[:8]}"
        self.experiment_id = mlflow.create_experiment(self.experiment_name)

    def test_new_experiment_gets_proxied_artifact_scheme_not_a_bare_path(self):
        # This is the exact assertion that would have caught the original bug:
        # a bare filesystem path here means writes bypass --serve-artifacts.
        experiment = mlflow.get_experiment(self.experiment_id)
        self.assertTrue(
            experiment.artifact_location.startswith("mlflow-artifacts:/"),
            msg=f"artifact_location={experiment.artifact_location!r} is not a proxied mlflow-artifacts:/ URI",
        )

    def test_logged_artifact_round_trips_through_the_tracking_server(self):
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id
            self.assertTrue(run.info.artifact_uri.startswith("mlflow-artifacts:/"))
            with tempfile.TemporaryDirectory() as tmp:
                probe_path = Path(tmp) / "probe.txt"
                probe_path.write_text("t093-regression-probe")
                mlflow.log_artifact(str(probe_path), "artifacts")

        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="artifacts/probe.txt",
        )
        self.assertEqual(Path(downloaded).read_text(), "t093-regression-probe")


if __name__ == "__main__":
    unittest.main()
