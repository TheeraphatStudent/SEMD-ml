import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

from core import settings
from infra import redis_client
from ml import prediction_service, training_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Errors caused by bad job input (missing/invalid fields) won't succeed on retry -- everything
# else (model download failures, transient MLflow/DB errors, etc.) might, given a retry or an
# infra fix, so it's surfaced as retryable.
NON_RETRYABLE_EXCEPTION_TYPES = (ValueError, TypeError, KeyError)


def build_job_failure_result(job_data: Dict[str, Any], job_type: str, exc: Exception) -> Dict[str, Any]:
    """Structured failure payload for a job that raised instead of completing.

    Deliberately excludes job_data itself (may carry a caller-supplied model_id/URL, but never
    echoes exception args beyond str(exc) to avoid leaking anything unexpected a deeper exception
    class might attach, e.g. connection objects with credentials in repr()).
    """
    return {
        'job_id': job_data.get('job_id'),
        'job_type': job_type,
        'status': 'failed',
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'failed_at': datetime.now(timezone.utc).isoformat(),
        'retryable': not isinstance(exc, NON_RETRYABLE_EXCEPTION_TYPES),
    }


class QueueWorker:

    def __init__(self):
        self.running = True
        self.training_queue = settings.training_queue
        self.prediction_queue = settings.prediction_queue
        self.result_queue = settings.result_queue

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        # Set the flag before logging: logging from inside a signal handler can raise
        # (e.g. a reentrant call into a stream the main thread was already writing to),
        # and if that happens after self.running = False, shutdown never gets flagged
        # and the worker loops forever, ignoring the signal.
        self.running = False
        logger.info(f"Received signal {signum}, shutting down gracefully...")

    def process_training_job(self, job_data: Dict[str, Any]):
        job_id = job_data.get('job_id', 'unknown')
        logger.info(f"Processing training job: {job_id}")

        try:
            result = training_service.execute_training(job_data)
            result['job_id'] = job_data.get('job_id')
            result['job_type'] = 'training'
            redis_client.push_to_queue(self.result_queue, result)
            logger.info("Training job completed, result pushed to queue")
        except Exception as exc:
            logger.error(f"Training job {job_id} failed: {exc}", exc_info=True)
            failure_result = build_job_failure_result(job_data, 'training', exc)
            redis_client.push_to_queue(self.result_queue, failure_result)
            logger.info(f"Training job {job_id} failure result pushed to queue")

    def process_prediction_job(self, job_data: Dict[str, Any]):
        job_id = job_data.get('job_id', 'unknown')
        logger.info(f"Processing prediction job: {job_id}")

        try:
            if 'urls' in job_data and isinstance(job_data['urls'], list):
                batch = prediction_service.batch_predict(job_data, input_source="queue")
                result = {
                    'status': 'success',
                    'results': [
                        {
                            'status': 'success',
                            'url': prediction['url'],
                            'prediction': prediction,
                            'model_id': prediction.get('model_version'),
                        }
                        for prediction in batch['predictions']
                    ],
                    'total': len(batch['predictions']),
                    'successful': len(batch['predictions']),
                    'failed': 0,
                }
            else:
                prediction = prediction_service.execute_prediction(job_data, input_source="queue")
                result = {
                    'status': 'success',
                    'url': prediction['url'],
                    'prediction': prediction,
                    'model_id': prediction.get('model_version'),
                }

            result['job_id'] = job_data.get('job_id')
            result['job_type'] = 'prediction'

            redis_client.push_to_queue(self.result_queue, result)
            logger.info("Prediction job completed, result pushed to queue")
        except Exception as exc:
            logger.error(f"Prediction job {job_id} failed: {exc}", exc_info=True)
            failure_result = build_job_failure_result(job_data, 'prediction', exc)
            redis_client.push_to_queue(self.result_queue, failure_result)
            logger.info(f"Prediction job {job_id} failure result pushed to queue")

    def start_training_worker(self):
        logger.info(
            f"Starting training worker, listening on queue: {self.training_queue}")

        while self.running:
            try:
                job_data = redis_client.pop_from_queue(
                    self.training_queue, timeout=5)

                if job_data:
                    self.process_training_job(job_data)

            except Exception as e:
                logger.error(
                    f"Error processing training job: {str(e)}", exc_info=True)
                time.sleep(1)

        logger.info('Training worker stopped')

    def start_prediction_worker(self):
        logger.info(
            f"Starting prediction worker, listening on queue: {self.prediction_queue}")

        while self.running:
            try:
                job_data = redis_client.pop_from_queue(
                    self.prediction_queue, timeout=5)

                if job_data:
                    self.process_prediction_job(job_data)

            except Exception as e:
                logger.error(
                    f"Error processing prediction job: {str(e)}", exc_info=True)
                time.sleep(1)

        logger.info('Prediction worker stopped')

    def start_combined_worker(self):
        logger.info('Starting combined worker for both training and prediction')
        logger.info(f"Training queue: {self.training_queue}")
        logger.info(f"Prediction queue: {self.prediction_queue}")

        while self.running:
            try:
                training_job = redis_client.pop_from_queue(
                    self.training_queue, timeout=1)
                if training_job:
                    self.process_training_job(training_job)
                    continue

                prediction_job = redis_client.pop_from_queue(
                    self.prediction_queue, timeout=1)
                if prediction_job:
                    self.process_prediction_job(prediction_job)
                    continue

                time.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Error in combined worker: {str(e)}", exc_info=True)
                time.sleep(1)

        logger.info('Combined worker stopped')


def main():
    logger.info('ML Service Queue Worker starting...')

    if not redis_client.ping():
        logger.error('Cannot connect to Redis. Please check Redis connection.')
        sys.exit(1)

    logger.info('Redis connection successful')

    worker = QueueWorker()

    import argparse
    parser = argparse.ArgumentParser(description='ML Service Queue Worker')
    parser.add_argument(
        '--mode',
        choices=['training', 'prediction', 'combined'],
        default='combined',
        help='Worker mode: training, prediction, or combined (default: combined)'
    )
    args = parser.parse_args()

    try:
        if args.mode == 'training':
            worker.start_training_worker()
        elif args.mode == 'prediction':
            worker.start_prediction_worker()
        else:
            worker.start_combined_worker()
    except KeyboardInterrupt:
        logger.info('Worker interrupted by user')
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
