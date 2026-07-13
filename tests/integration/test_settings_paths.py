"""Regression coverage for the env-var -> settings -> path resolution chain.

Backfilled after T-092/T-093: MLServiceSettings previously defaulted
MLFLOW_ARTIFACT_ROOT to a cwd-relative path ("./artifacts/mlflow"), which
resolved differently depending on whether the process launched from the repo
root, from src/, or inside a container (WORKDIR /app/src) -- silently
producing an artifact_location no other process could read back. These tests
pin the settings contract so a future change can't reintroduce that class of
bug without a red test.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from core.config import PROJECT_ROOT, MLServiceSettings


class SettingsPathResolutionTests(unittest.TestCase):
    def test_project_root_is_repo_root_not_src(self):
        # config.py lives at src/core/config.py; PROJECT_ROOT must be two levels up.
        self.assertTrue((PROJECT_ROOT / "src" / "core" / "config.py").is_file())
        self.assertTrue((PROJECT_ROOT / "makefile").is_file())

    def test_dataset_and_models_paths_are_absolute(self):
        settings = MLServiceSettings()
        for attr in ("dataset_path", "extraction_path", "models_path", "reports_path"):
            with self.subTest(attr=attr):
                self.assertTrue(Path(getattr(settings, attr)).is_absolute())

    def test_mlflow_artifact_root_default_is_proxied_scheme_not_a_bare_relative_path(self):
        settings = MLServiceSettings()
        # Must be either an explicit mlflow-artifacts:/ URI (proxied through the tracking
        # server) or an absolute path. A bare relative path silently resolves against
        # whatever the process cwd happens to be at experiment-creation time.
        root = settings.mlflow_artifact_root
        self.assertTrue(
            root.startswith("mlflow-artifacts:/") or Path(root).is_absolute(),
            msg=f"MLFLOW_ARTIFACT_ROOT={root!r} is a bare relative path; see docs/section-10-infrastructure-validation.md",
        )

    def test_redis_password_field_is_env_overridable(self):
        settings = MLServiceSettings(redis_password="probe-secret")
        self.assertEqual(settings.redis_password, "probe-secret")

    def test_env_file_resolves_to_project_root_dotenv(self):
        settings = MLServiceSettings()
        self.assertEqual(Path(settings.Config.env_file), PROJECT_ROOT / ".env")


if __name__ == "__main__":
    unittest.main()
