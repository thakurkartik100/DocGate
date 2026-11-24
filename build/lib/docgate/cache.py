import time
from collections import OrderedDict


class TTLCache:
    """Small LRU+TTL cache for repeated queries. Single-process only (shortcut: use Redis when scaling out)."""

    def __init__(self, size: int = 512, ttl: float = 300.0):
        self.size, self.ttl, self.data = size, ttl, OrderedDict()

    def get(self, key):
        item = self.data.get(key)
        if item is None or time.monotonic() - item[0] > self.ttl:
            self.data.pop(key, None)
            return None
        self.data.move_to_end(key)
        return item[1]

    def put(self, key, value):
        self.data[key] = (time.monotonic(), value)
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)
