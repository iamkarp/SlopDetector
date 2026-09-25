"""Content-hash keyed local cache so re-scanning unchanged text never re-bills."""

from __future__ import annotations

import hashlib
import json
import os


def _key(model: str, text: str) -> str:
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
        path = os.path.join(self.cache_dir, _key(model, text) + ".json")
        if os.path.isfile(path):
            try:
                return json.loads(open(path, encoding="utf-8").read())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def set(self, model: str, text: str, value: dict) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, _key(model, text) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f)
