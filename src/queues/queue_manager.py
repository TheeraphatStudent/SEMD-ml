import json
import logging
from typing import Dict, List, Any

from core import get_logger, settings
from infra import redis_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueueManager:

    def __init__(self):
        self.queues = {
            'training': settings.training_queue,
            'prediction': settings.prediction_queue,
            'result': settings.result_queue
        }

    def get_queue_status(self) -> Dict[str, Any]:
        try:
            status = {}

            for name, queue in self.queues.items():
                length = redis_client.client.llen(queue)
                jobs = []

                if length > 0:
                    job_items = redis_client.client.lrange(
                        queue, 0, length - 1)
                    for i, job in enumerate(job_items):
                        try:
                            job_data = json.loads(job.decode('utf-8'))
                            jobs.append({
                                'index': i + 1,
                                'data': job_data
                            })
                        except Exception as e:
                            jobs.append({
                                'index': i + 1,
                                'data': str(job),
                                'error': str(e)
                            })

                status[name] = {
                    'queue_name': queue,
                    'job_count': length,
                    'jobs': jobs
                }

            return status

        except Exception as e:
            logger.error(f"Failed to get queue status: {str(e)}")
            raise

    def print_queue_status(self, status: Dict[str, Any]) -> None:
        print(f"{'-'*50}-\n")

        for name, info in status.items():
            print(
                f"{name.capitalize()} queue ({info['queue_name']}): {info['job_count']} jobs")
            if info['jobs']:
                for job in info['jobs']:
                    if 'error' in job:
                        print(
                            f"  Job {job['index']}: {job['data']} (error parsing: {job['error']})")
                    else:
                        print(
                            f"  Job {job['index']}: {json.dumps(job['data'], indent=2)}")

        print(f"\n{'-'*50}-")
