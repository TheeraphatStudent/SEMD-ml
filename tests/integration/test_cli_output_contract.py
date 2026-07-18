"""T11 (docs/refactoring-plan.md): lock CLI JSON output contracts before any future
refactor touches the command implementations.

These assert the *current, real* shapes emitted by `main.py predict`/`predict-test`
and by the queue worker's prediction-result payload (see tests/unit/test_queue_worker.py
for the worker side) -- not the aspirational shape sketched in refactoring-plan.md's
"Interfaces to preserve" section, which predates this response format. If the shape
changes intentionally, update the locked key set here in the same change.

Uses direct calls into cli.commands.predict with a stubbed PredictionService instead of
a live subprocess: predict now requires a reachable MLflow registry (see model_registry.py),
which CI/local test runs cannot assume. Stubbing the service still exercises the real
CLI argument handling and emit_result() JSON serialization path.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from cli.commands.predict import cmd_predict, cmd_predict_test

SINGLE_PREDICTION_RESPONSE = {
    "url": "http://phishy-bad0.example.net/login/verify?next=x",
    "prediction": "benign",
    "is_malicious": False,
    "confidence": 0.6353361537795557,
    "feature_schema_version": "2.1.0",
    "prediction_time_ms": 19.32,
    "model_name": "semd-malicious-url-detector",
    "model_version": "5",
    "model_alias": "champion",
    "prediction_id": "dfcc1ffb3cd9486ab10f8482818a9aa9",
}

LOCKED_PREDICTION_KEYS = {
    "url",
    "prediction",
    "is_malicious",
    "confidence",
    "feature_schema_version",
    "prediction_time_ms",
    "model_name",
    "model_version",
    "model_alias",
    "prediction_id",
}


def _run_predict(args: argparse.Namespace) -> dict:
    buf = io.StringIO()
    with patch("cli.commands.predict.prediction_service") as mock_service:
        mock_service.execute_prediction.return_value = dict(SINGLE_PREDICTION_RESPONSE)
        with contextlib.redirect_stdout(buf):
            exit_code = cmd_predict(args)
    return exit_code, json.loads(buf.getvalue())


class PredictJsonContractTests(unittest.TestCase):
    def test_single_url_output_has_locked_key_set(self):
        args = argparse.Namespace(url="http://phishy-bad0.example.net/login/verify?next=x",
                                   urls=None, url_file=None, model_id=None, output=None)
        exit_code, payload = _run_predict(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(set(payload.keys()), LOCKED_PREDICTION_KEYS)
        self.assertEqual(payload["is_malicious"], False)
        self.assertIsInstance(payload["confidence"], float)

    def test_multiple_urls_wrap_in_predictions_key(self):
        args = argparse.Namespace(url=None, urls=["https://a.example.com", "https://b.example.com"],
                                   url_file=None, model_id=None, output=None)
        with patch("cli.commands.predict.prediction_service") as mock_service:
            mock_service.execute_prediction.return_value = dict(SINGLE_PREDICTION_RESPONSE)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exit_code = cmd_predict(args)
            payload = json.loads(buf.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(set(payload.keys()), {"predictions"})
        self.assertEqual(len(payload["predictions"]), 2)
        self.assertEqual(set(payload["predictions"][0].keys()), LOCKED_PREDICTION_KEYS)

    def test_no_url_provided_returns_nonzero_without_raising(self):
        args = argparse.Namespace(url=None, urls=None, url_file=None, model_id=None, output=None)
        with patch("cli.commands.predict.prediction_service"):
            exit_code = cmd_predict(args)
        self.assertEqual(exit_code, 1)


class PredictTestJsonContractTests(unittest.TestCase):
    def test_batch_predict_output_has_locked_predictions_key(self):
        args = argparse.Namespace(url="https://a.example.com", urls=None, csv=None,
                                   model_id=None, output=None)
        buf = io.StringIO()
        with patch("cli.commands.predict.prediction_service") as mock_service:
            mock_service.batch_predict.return_value = {"predictions": [dict(SINGLE_PREDICTION_RESPONSE)]}
            with contextlib.redirect_stdout(buf):
                exit_code = cmd_predict_test(args)
            payload = json.loads(buf.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(set(payload.keys()), {"predictions"})
        self.assertEqual(set(payload["predictions"][0].keys()), LOCKED_PREDICTION_KEYS)


if __name__ == "__main__":
    unittest.main()
