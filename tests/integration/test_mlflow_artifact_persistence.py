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
from tracking.mlflow_tracker import MLflowTracker, UnsafeExperimentArtifactLocationError

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

    def test_mlflow_tracker_class_itself_gets_the_proxied_scheme_for_a_new_experiment(self):
        # The two tests above exercise raw mlflow calls. This one exercises the project's own
        # MLflowTracker (mlflow_tracker.py), which is what training_service.py actually calls --
        # confirming the *code path*, not just the underlying mlflow SDK behavior, was fixed.
        tracker = MLflowTracker()
        tracker.experiment_name = self.experiment_name
        run_id = tracker.start_run(run_name="t093-tracker-probe")
        self.assertIsNotNone(run_id, msg=tracker.last_error)
        artifact_uri = mlflow.active_run().info.artifact_uri
        tracker.end_run()
        self.assertTrue(
            artifact_uri.startswith("mlflow-artifacts:/"),
            msg=f"MLflowTracker produced artifact_uri={artifact_uri!r}, not a proxied URI",
        )


@unittest.skipUnless(_mlflow_server_reachable(), "Live MLflow tracking server not reachable")
class MlflowExperimentReuseSafetyGuardTests(unittest.TestCase):
    """Regression coverage for the fix to Remaining Blocker #1 (see
    docs/section-10-infrastructure-validation.md): MLflowTracker only assigns the proxied
    mlflow-artifacts:/ scheme when IT creates the experiment. artifact_location is permanent, set
    once at creation -- reusing an experiment that predates the proxy fix (like the real
    semd-url-classification experiment) used to silently keep training into a bare filesystem
    path forever, reproducing the exact bug T-093 set out to fix.

    This class previously documented that silent-reuse behavior as a known gap. It now asserts
    the fixed behavior: reusing such an experiment raises UnsafeExperimentArtifactLocationError
    instead of silently continuing.
    """

    def setUp(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.experiment_name = f"t093-reuse-gap-{uuid.uuid4().hex[:8]}"
        # Simulate a pre-fix experiment: an explicit bare filesystem path, exactly what the old
        # _normalize_artifact_root used to produce.
        self.legacy_artifact_location = f"/tmp/t093-legacy-{uuid.uuid4().hex[:8]}"
        self.experiment_id = mlflow.create_experiment(
            self.experiment_name, artifact_location=self.legacy_artifact_location
        )

    def test_reusing_an_existing_bare_path_experiment_fails_loudly(self):
        tracker = MLflowTracker()
        tracker.experiment_name = self.experiment_name
        with self.assertRaises(UnsafeExperimentArtifactLocationError) as ctx:
            tracker.start_run(run_name="t093-reuse-gap-probe")
        message = str(ctx.exception)
        self.assertIn(self.experiment_name, message)
        self.assertIn(self.legacy_artifact_location, message)
        # No run should have been left dangling against the unsafe experiment.
        self.assertIsNone(mlflow.active_run())
        # The tracker must not have silently disabled itself either -- it's the specific-error
        # path, not the generic "MLflow unreachable" path.
        self.assertTrue(tracker.enabled)

    def test_safe_new_experiment_still_works_after_guard_added(self):
        # The guard must not false-positive on an experiment it creates itself.
        tracker = MLflowTracker()
        tracker.experiment_name = f"t093-reuse-gap-safe-{uuid.uuid4().hex[:8]}"
        run_id = tracker.start_run(run_name="t093-reuse-gap-safe-probe")
        self.assertIsNotNone(run_id, msg=tracker.last_error)
        artifact_uri = mlflow.active_run().info.artifact_uri
        tracker.end_run()
        self.assertTrue(artifact_uri.startswith("mlflow-artifacts:/"))


if __name__ == "__main__":
    unittest.main()
