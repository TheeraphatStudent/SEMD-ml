"""Regression coverage for T-092 (Redis authentication alignment).

Requires a live Redis reachable via REDIS_HOST/REDIS_PORT (matching whatever
.env / the environment currently point at) with a password configured, i.e.
`podman compose -f database/docker-compose.database.yaml up -d` in
semd-backend. Skips itself entirely if that Redis is unreachable, since this
section explicitly should not require the full stack to be running for the
rest of the suite to pass.
"""

from __future__ import annotations

import unittest

import redis as redis_lib

from core.config import MLServiceSettings

settings = MLServiceSettings()


def _redis_reachable() -> bool:
    try:
        client = redis_lib.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            socket_connect_timeout=2,
        )
        return bool(client.ping())
    except Exception:
        return False


@unittest.skipUnless(_redis_reachable(), "Live Redis not reachable with current .env credentials")
class RedisAuthenticationTests(unittest.TestCase):
    def test_configured_credentials_authenticate(self):
        client = redis_lib.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            socket_connect_timeout=2,
        )
        self.assertTrue(client.ping())

    def test_wrong_password_is_rejected_not_silently_accepted(self):
        if not settings.redis_password:
            self.skipTest("Redis has no password configured; nothing to reject")
        client = redis_lib.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password="definitely-the-wrong-password",
            db=settings.redis_db,
            socket_connect_timeout=2,
        )
        with self.assertRaises(redis_lib.exceptions.AuthenticationError):
            client.ping()

    def test_push_and_pop_round_trip_on_prediction_queue(self):
        client = redis_lib.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        probe_queue = "ml_prediction_queue_test_probe"
        client.delete(probe_queue)
        client.lpush(probe_queue, '{"probe": true}')
        _, payload = client.brpop(probe_queue, timeout=2)
        self.assertEqual(payload, '{"probe": true}')


if __name__ == "__main__":
    unittest.main()
