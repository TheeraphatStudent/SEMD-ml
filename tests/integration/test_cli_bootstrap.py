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
        for subcommand in ["train", "predict", "worker", "queue-status", "evaluate"]:
            with self.subTest(subcommand=subcommand):
                result = run_in_src(["main.py", subcommand, "--help"])
                self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
