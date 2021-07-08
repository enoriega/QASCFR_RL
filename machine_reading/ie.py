from typing import Set

import redis


class RedisWrapper:

    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379)

    # TODO add lru_cache here
    def get_extractions(self, text: str) -> Set[bytes]:
        extractions = self.r.smembers(f"ph#{text}")
        return extractions
