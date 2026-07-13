from typing import Any

from core import get_logger
from queues import QueueManager
from workers import QueueWorker

logger = get_logger(__name__)


def cmd_worker(args: Any) -> int:
    logger.info('Starting queue worker...')

    worker = QueueWorker()

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
        return 1


def cmd_queue_status(args: Any) -> int:
    logger.info('Checking Redis queue status...')

    try:
        manager = QueueManager()
        status = manager.get_queue_status()
        manager.print_queue_status(status)
        return 0
    except Exception as e:
        logger.error(f"Failed to check queue status: {str(e)}")
        return 1
