"""Clean-environment regression coverage for the CLI import chain and entrypoints.

These previously slipped past the unit suite because unit tests import modules
directly inside the already-running pytest process (sys.path already has src/ on
it, .env already loaded). Running main.py / verify_imports.py as a fresh
subprocess from the project root reproduces what a clean checkout or a fresh
container actually experiences.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


def run_in_src(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class CliBootstrapTests(unittest.TestCase):
    def test_main_help_exits_zero(self):
        result = run_in_src(["main.py", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage", result.stdout.lower())

    def test_main_help_does_not_require_live_redis_or_mlflow(self):
        # --help must work even when no infrastructure is running: it should never
        # eagerly construct RedisClient/MLflowTracker connections just to print usage.
        result = run_in_src(
            ["main.py", "--help"],
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_verify_imports_script_exits_zero(self):
        result = run_in_src(["verify_imports.py"], timeout=120)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_subcommand_help_exits_zero_for_each_group(self):
        # T11 (docs/refactoring-plan.md): lock --help for every subcommand main.py
        # actually registers (see src/cli/main.py), not just a handful, so a future
        # CLI move can diff this list against argparse instead of guessing coverage.
        subcommands = [
            "data", "train", "train-obo", "evaluate", "predict", "register",
            "promote", "rollback", "gate-check", "feedback", "review", "monitor",
            "retrain", "predict-test", "feature-engineering", "worker",
            "queue-status", "data-migrate", "data-migrate-feature",
        ]
        for subcommand in subcommands:
            with self.subTest(subcommand=subcommand):
                result = run_in_src(["main.py", subcommand, "--help"])
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage", result.stdout.lower())

    def test_nested_data_subcommand_help_exits_zero(self):
        result = run_in_src(["main.py", "data", "validate", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage", result.stdout.lower())

    def test_registered_subcommands_match_locked_list(self):
        # If someone adds/removes a subcommand in cli/main.py without updating the
        # locked list above, this fails loudly instead of the coverage silently drifting.
        result = run_in_src(["main.py", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        locked = {
            "data", "train", "train-obo", "evaluate", "predict", "register",
            "promote", "rollback", "gate-check", "feedback", "review", "monitor",
            "retrain", "predict-test", "feature-engineering", "worker",
            "queue-status", "data-migrate", "data-migrate-feature",
        }
        for name in locked:
            self.assertIn(name, result.stdout, msg=f"'{name}' missing from `main.py --help` output")

    def test_cli_bootstrap_imports_without_running_from_src(self):
        # cli must be importable via the editable install regardless of cwd, not
        # only when src/ happens to be on sys.path via script-directory insertion.
        result = subprocess.run(
            [sys.executable, "-c", "import cli.bootstrap; print(cli.bootstrap.__file__)"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("bootstrap", result.stdout)


if __name__ == "__main__":
    unittest.main()
