"""Thread-safety tests for ucat.config.Settings.

The async verify worker reads `settings.get("llm")` etc. from a background
thread while the UI thread mutates settings via `settings.set(...)` and saves
to disk. Without locking, the underlying dict can tear mid-write or two saves
can interleave producing a corrupt JSON file.

These tests stress that path: many concurrent reads + writes against a single
Settings instance, then verify the on-disk JSON is valid and consistent.
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ucat.config import Settings  # noqa: E402


def test_concurrent_writes_do_not_corrupt_json():
    """Many threads spamming `set()` simultaneously should leave the on-disk
    file in a parseable state — never half-written or with mixed-version keys."""
    path = tempfile.mktemp(suffix=".json")
    try:
        s = Settings(path=path)

        N_THREADS = 16
        N_WRITES_PER_THREAD = 50
        barrier = threading.Barrier(N_THREADS)

        def worker(tid: int):
            barrier.wait()
            for i in range(N_WRITES_PER_THREAD):
                s.set("bulk_quantity", tid * 1000 + i)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # On-disk JSON must be parseable. Without the lock, a torn write
        # would leave invalid JSON behind.
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        assert "bulk_quantity" in saved
        assert isinstance(saved["bulk_quantity"], int)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_concurrent_reads_during_writes_never_see_mid_update_state():
    """A reader thread loops calling `get()` while writers mutate settings.
    With the lock, reads either see the old or new value, never a torn dict."""
    path = tempfile.mktemp(suffix=".json")
    try:
        s = Settings(path=path)
        s.set("bulk_quantity", 0)

        observed_values = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                v = s.get("bulk_quantity")
                # Lock contract: every read returns an int (the type the
                # writers set). If a write was torn we'd see None or some
                # half-state object.
                assert isinstance(v, int), f"got non-int: {v!r}"
                observed_values.append(v)

        def writer(tid: int):
            for i in range(100):
                s.set("bulk_quantity", tid * 1000 + i)

        r = threading.Thread(target=reader, daemon=True)
        ws = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        r.start()
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        stop.set()
        r.join(timeout=2.0)

        assert len(observed_values) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_concurrent_writers_disk_matches_final_memory_state():
    """Regression for the lock-release-before-save bug. With the bug,
    threads A and B can both mutate, release the lock, then race to save —
    so the disk file may reflect a different value than the in-memory dict.
    With the fix (save inside the lock), the last writer's value is on disk
    AND in memory; they can never diverge."""
    path = tempfile.mktemp(suffix=".json")
    try:
        s = Settings(path=path)

        N_THREADS = 8
        N_WRITES = 200

        def worker(tid: int):
            for i in range(N_WRITES):
                s.set("bulk_quantity", tid * 10000 + i)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all writes settle, disk and memory MUST agree on the
        # last-written value. Bug symptom: they diverge.
        in_memory = s.get("bulk_quantity")
        with open(path, encoding="utf-8") as f:
            on_disk = json.load(f)["bulk_quantity"]
        assert in_memory == on_disk, (
            f"disk/memory drift: memory={in_memory}, disk={on_disk} — "
            "the lock-release-before-save bug is back"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_set_then_get_is_visible_to_other_threads():
    """Memory-visibility check: a value set on thread A is observable by
    thread B via `get()`. The lock acts as a memory barrier in CPython, but
    we test the contract explicitly."""
    path = tempfile.mktemp(suffix=".json")
    try:
        s = Settings(path=path)
        s.set("bulk_quantity", -1)

        observed_in_thread_b = []

        def thread_a():
            s.set("bulk_quantity", 42)

        def thread_b():
            t.join()  # wait for thread_a to finish
            observed_in_thread_b.append(s.get("bulk_quantity"))

        t = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t.start()
        t2.start()
        t.join()
        t2.join()

        assert observed_in_thread_b == [42]
    finally:
        if os.path.exists(path):
            os.unlink(path)
