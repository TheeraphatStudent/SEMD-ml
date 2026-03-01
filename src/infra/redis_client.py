import json
from typing import Optional, Any, Dict
import redis

from core import settings


class RedisClient:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'client'):
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password if settings.redis_password else None,
                db=settings.redis_db,
                decode_responses=True
            )

    def push_to_queue(self, queue_name: str, data: Dict[str, Any]) -> int:
        return self.client.lpush(queue_name, json.dumps(data))

    def pop_from_queue(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        result = self.client.brpop(queue_name, timeout=timeout)
        if result:
            _, data = result
            return json.loads(data)
        return None

    def get_cache(self, key: str) -> Optional[Any]:
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def set_cache(self, key: str, data: Any, ttl: int = 3600) -> bool:
        return self.client.setex(key, ttl, json.dumps(data))

    def delete_cache(self, key: str) -> int:
        return self.client.delete(key)

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False


redis_client = RedisClient()
