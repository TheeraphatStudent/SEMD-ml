import time
import signal
import sys
import logging
from typing import Dict, Any

from core import settings, get_logger
from infra import redis_client
from ml import training_service, prediction_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueueWorker:

    def __init__(self):
        self.running = True
        self.training_queue = settings.training_queue
        self.prediction_queue = settings.prediction_queue
        self.result_queue = settings.result_queue

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def process_training_job(self, job_data: Dict[str, Any]):
        logger.info(
            f"Processing training job: {job_data.get('job_id', 'unknown')}")

        result = training_service.execute_training(job_data)

        result['job_id'] = job_data.get('job_id')
        result['job_type'] = 'training'

        redis_client.push_to_queue(self.result_queue, result)
        logger.info(f"Training job completed, result pushed to queue")

    def process_prediction_job(self, job_data: Dict[str, Any]):
        logger.info(
            f"Processing prediction job: {job_data.get('job_id', 'unknown')}")

        if 'urls' in job_data and isinstance(job_data['urls'], list):
            result = prediction_service.batch_predict(job_data)
        else:
            result = prediction_service.execute_prediction(job_data)

        result['job_id'] = job_data.get('job_id')
        result['job_type'] = 'prediction'

        redis_client.push_to_queue(self.result_queue, result)
        logger.info(f"Prediction job completed, result pushed to queue")

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
