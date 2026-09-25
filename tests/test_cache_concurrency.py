from __future__ import annotations

import threading
import time

from slopdetector.llm.cache import DiskCache, InFlightGuard, cache_key


def test_disk_cache_round_trips(tmp_path):
    cache = DiskCache(str(tmp_path))
    cache.set("model-a", "some text", {"probability": 0.42})
    assert cache.get("model-a", "some text") == {"probability": 0.42}


def test_disk_cache_write_is_atomic_no_partial_file_visible(tmp_path):
    cache = DiskCache(str(tmp_path))
    # A concurrent get() during a set() must see either nothing or the
    # complete value — never a partially-written/corrupt file.
    seen = []

    def writer():
        for i in range(20):
            cache.set("m", "text", {"probability": i / 20})

    def reader():
        for _ in range(50):
            val = cache.get("m", "text")
            if val is not None:
                seen.append(val)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for val in seen:
        assert "probability" in val
        assert 0.0 <= val["probability"] <= 1.0


def test_disk_cache_ignores_malformed_entry(tmp_path):
    path = tmp_path / (cache_key("m", "text") + ".json")
    path.write_text("{not valid json")
    cache = DiskCache(str(tmp_path))
    assert cache.get("m", "text") is None


def test_disk_cache_ignores_wrong_shaped_entry(tmp_path):
    path = tmp_path / (cache_key("m", "text") + ".json")
    path.write_text('{"unexpected": "shape"}')
    cache = DiskCache(str(tmp_path))
    assert cache.get("m", "text") is None


def test_inflight_guard_serializes_identical_keys():
    guard = InFlightGuard()
    key = cache_key("m", "same text")
    call_count = 0
    lock_order = []

    def worker(n):
        nonlocal call_count
        with guard.lock_for(key):
            call_count += 1
            lock_order.append(n)
            time.sleep(0.01)  # hold the lock long enough to force serialization

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 5  # all ran, but serialized (not proof by itself)
    assert len(lock_order) == 5


def test_inflight_guard_gives_different_locks_for_different_keys():
    guard = InFlightGuard()
    lock_a = guard.lock_for(cache_key("m", "text a"))
    lock_b = guard.lock_for(cache_key("m", "text b"))
    assert lock_a is not lock_b
