"""T10 (docs/refactoring-plan.md): unit coverage for QueueManager.get_queue_status's
job parsing. Regression test for a decode bug fixed this session: redis_client is
configured with decode_responses=True (see infra/redis_client.py), so lrange
already returns str, not bytes -- calling job.decode('utf-8') unconditionally
raised 'str' object has no attribute 'decode' and silently degraded every queued
job to an {"error": ...} entry instead of parsing it.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from queues.queue_manager import QueueManager


class QueueStatusParsingTests(unittest.TestCase):
    def setUp(self):
        self.manager = QueueManager()

    def _status_for(self, queue_items: list):
        fake_client = MagicMock()
        fake_client.llen.return_value = len(queue_items)
        fake_client.lrange.return_value = queue_items
        with patch("queues.queue_manager.redis_client") as mock_redis:
            mock_redis.client = fake_client
            return self.manager.get_queue_status()

    def test_str_job_payload_parses_without_decode_error(self):
        # decode_responses=True means lrange returns str, not bytes -- this is the
        # real shape redis-py hands back in this codebase.
        payload = json.dumps({"job_id": "abc", "url": "https://example.com"})
        status = self._status_for([payload])

        job = status["training"]["jobs"][0]
        self.assertNotIn("error", job)
        self.assertEqual(job["data"], {"job_id": "abc", "url": "https://example.com"})

    def test_bytes_job_payload_still_parses(self):
        # Defensive: still handle bytes correctly if the redis client were ever
        # reconfigured without decode_responses=True.
        payload = json.dumps({"job_id": "xyz"}).encode("utf-8")
        status = self._status_for([payload])

        job = status["training"]["jobs"][0]
        self.assertNotIn("error", job)
        self.assertEqual(job["data"], {"job_id": "xyz"})

    def test_malformed_json_reports_error_without_raising(self):
        status = self._status_for(["not valid json"])

        job = status["training"]["jobs"][0]
        self.assertIn("error", job)
        self.assertEqual(job["data"], "not valid json")

    def test_empty_queue_reports_zero_jobs(self):
        status = self._status_for([])
        self.assertEqual(status["training"]["job_count"], 0)
        self.assertEqual(status["training"]["jobs"], [])

    def test_all_three_queues_present_in_status(self):
        status = self._status_for([])
        self.assertEqual(set(status.keys()), {"training", "prediction", "result"})


if __name__ == "__main__":
    unittest.main()
