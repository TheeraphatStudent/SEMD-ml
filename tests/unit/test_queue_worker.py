"""Regression coverage for Remaining Blocker #4 (silent queue job loss).

Before this fix: a raised exception inside process_prediction_job/process_training_job
propagated straight to the outer worker-loop's `except Exception`, which only logs and sleeps.
Because Redis's BRPOP already destructively removed the job from the input queue by that point,
the job simply vanished -- no retry, no dead-letter, no result. See
docs/section-10-infrastructure-validation.md.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tracking.model_registry import ModelRegistryError
from workers.queue_worker import QueueWorker, build_job_failure_result


class BuildJobFailureResultTests(unittest.TestCase):
    def test_value_error_is_not_retryable(self):
        result = build_job_failure_result({'job_id': 'j1'}, 'prediction', ValueError('No URL provided'))
        self.assertEqual(result['job_id'], 'j1')
        self.assertEqual(result['job_type'], 'prediction')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error_type'], 'ValueError')
        self.assertEqual(result['error_message'], 'No URL provided')
        self.assertFalse(result['retryable'])
        self.assertIn('failed_at', result)

    def test_model_registry_error_is_retryable(self):
        result = build_job_failure_result(
            {'job_id': 'j2'}, 'prediction', ModelRegistryError('Unable to load model from MLflow registry')
        )
        self.assertEqual(result['error_type'], 'ModelRegistryError')
        self.assertTrue(result['retryable'])


class ProcessPredictionJobFailureTests(unittest.TestCase):
    def setUp(self):
        self.worker = QueueWorker.__new__(QueueWorker)  # skip __init__ (signal handlers)
        self.worker.result_queue = 'ml_result_queue'

    def test_model_not_found_publishes_structured_failure_not_silently_lost(self):
        with patch('workers.queue_worker.prediction_service') as mock_pred, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_pred.execute_prediction.side_effect = ModelRegistryError(
                "Unable to load model from MLflow registry: Failed to download artifacts"
            )
            self.worker.process_prediction_job({'job_id': 'job-404', 'url': 'https://example.com'})

            mock_redis.push_to_queue.assert_called_once()
            queue_name, payload = mock_redis.push_to_queue.call_args[0]
            self.assertEqual(queue_name, 'ml_result_queue')
            self.assertEqual(payload['job_id'], 'job-404')
            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['error_type'], 'ModelRegistryError')
            self.assertTrue(payload['retryable'])

    def test_invalid_payload_missing_url_publishes_structured_failure(self):
        with patch('workers.queue_worker.prediction_service') as mock_pred, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_pred.execute_prediction.side_effect = ValueError("No URL provided for prediction")
            self.worker.process_prediction_job({'job_id': 'job-bad-payload'})

            payload = mock_redis.push_to_queue.call_args[0][1]
            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['error_type'], 'ValueError')
            self.assertFalse(payload['retryable'])

    def test_generic_prediction_exception_publishes_structured_failure(self):
        with patch('workers.queue_worker.prediction_service') as mock_pred, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_pred.execute_prediction.side_effect = RuntimeError("feature extraction blew up")
            self.worker.process_prediction_job({'job_id': 'job-boom', 'url': 'https://example.com'})

            payload = mock_redis.push_to_queue.call_args[0][1]
            self.assertEqual(payload['error_type'], 'RuntimeError')
            self.assertTrue(payload['retryable'])

    def test_successful_job_still_pushes_success_result_not_failure(self):
        with patch('workers.queue_worker.prediction_service') as mock_pred, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_pred.execute_prediction.return_value = {
                'url': 'https://example.com',
                'prediction': 'benign',
                'model_version': '4',
            }
            self.worker.process_prediction_job({'job_id': 'job-ok', 'url': 'https://example.com'})

            payload = mock_redis.push_to_queue.call_args[0][1]
            self.assertEqual(payload['status'], 'success')
            self.assertNotIn('error_type', payload)

    def test_successful_job_result_shape_is_locked(self):
        # T11 (docs/refactoring-plan.md): this is the shape semd-backend polls via
        # ml_result_queue -- must stay {status, url, prediction, model_id, job_id,
        # job_type} until it is explicitly versioned. See test_cli_output_contract.py
        # for the separate, unrelated `main.py predict` CLI output contract.
        with patch('workers.queue_worker.prediction_service') as mock_pred, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_pred.execute_prediction.return_value = {
                'url': 'https://example.com',
                'prediction': 'benign',
                'model_version': '4',
            }
            self.worker.process_prediction_job({'job_id': 'job-shape', 'url': 'https://example.com'})

            payload = mock_redis.push_to_queue.call_args[0][1]
            self.assertEqual(set(payload.keys()), {'status', 'url', 'prediction', 'model_id', 'job_id', 'job_type'})
            self.assertEqual(payload['job_type'], 'prediction')
            self.assertEqual(payload['model_id'], '4')

    def test_worker_continues_processing_later_jobs_after_one_failure(self):
        with patch('workers.queue_worker.prediction_service') as mock_pred, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_pred.execute_prediction.side_effect = [
                ModelRegistryError("download failed"),
                {'url': 'https://example.com/2', 'prediction': 'benign', 'model_version': '4'},
            ]

            self.worker.process_prediction_job({'job_id': 'job-1', 'url': 'https://example.com/1'})
            self.worker.process_prediction_job({'job_id': 'job-2', 'url': 'https://example.com/2'})

            self.assertEqual(mock_redis.push_to_queue.call_count, 2)
            first_payload = mock_redis.push_to_queue.call_args_list[0][0][1]
            second_payload = mock_redis.push_to_queue.call_args_list[1][0][1]
            self.assertEqual(first_payload['status'], 'failed')
            self.assertEqual(second_payload['status'], 'success')


class ProcessTrainingJobFailureTests(unittest.TestCase):
    def setUp(self):
        self.worker = QueueWorker.__new__(QueueWorker)
        self.worker.result_queue = 'ml_result_queue'

    def test_training_exception_publishes_structured_failure_not_silently_lost(self):
        with patch('workers.queue_worker.training_service') as mock_train, \
             patch('workers.queue_worker.redis_client') as mock_redis:
            mock_train.execute_training.side_effect = ValueError("Dataset cleaning removed all rows")
            self.worker.process_training_job({'job_id': 'train-1', 'dataset_files': ['bad.csv']})

            mock_redis.push_to_queue.assert_called_once()
            payload = mock_redis.push_to_queue.call_args[0][1]
            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['job_type'], 'training')
            self.assertEqual(payload['error_type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
