"""Content-hash keyed local cache so re-scanning unchanged text never re-bills.

Writes are atomic (temp file + os.replace) so a reader under the
ThreadPoolExecutor never observes a partially-written file. InFlightGuard
additionally deduplicates concurrent identical (model, text) requests so
two threads scoring the same paragraph don't both hit the API.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading


def cache_key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class DiskCache:
    def __init__(self, cache_dir: str | None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, model: str, text: str) -> dict | None:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, cache_key(model, text) + ".json")
        if os.path.isfile(path):
            try:
                value = json.loads(open(path, encoding="utf-8").read())
            except (json.JSONDecodeError, OSError):
                return None
            if not isinstance(value, dict) or "probability" not in value:
                return None  # ignore malformed/stale entries rather than crash
            return value
        return None

    def set(self, model: str, text: str, value: dict) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, cache_key(model, text) + ".json")
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f)
            os.replace(tmp_path, path)  # atomic on POSIX and Windows
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


class InFlightGuard:
    """Per-key locks so concurrent threads scoring the same (model, text)
    only make one API call; the rest block, then read the cache."""

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def lock_for(self, key: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock
