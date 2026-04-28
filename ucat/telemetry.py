"""Structured telemetry — per-run JSONL traces + Python logging.

Every generation, verification, and indexing run emits a structured event with
section, latency, token usage, cost, retrieved IDs, verdict, coverage tags, and
predicted difficulty. This is what powers the analytics dashboard and the
nightly benchmark loop.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from .config import TELEMETRY_FILE

# ─── Python logger ────────────────────────────────────────────────────────────

_LOG_LEVEL = os.environ.get("UCAT_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("ucat")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

# ─── JSONL trace store ────────────────────────────────────────────────────────

class Telemetry:
    """Append-only JSONL writer. One line = one event."""

    def __init__(self, path: str = TELEMETRY_FILE):
        self.path = path
        self._fh = None

    def _file(self):
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        return self._fh

    def emit(self, event: str, **fields: Any):
        rec: Dict[str, Any] = {
            "ts":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        # Drop fields that JSON can't represent natively.
        try:
            line = json.dumps(rec, default=str, ensure_ascii=False)
            self._file().write(line + "\n")
        except Exception as e:
            logger.warning("Telemetry write failed: %s", e)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# Module-level singleton — convenient and safe for a desktop app.
_TELEMETRY = Telemetry()


def emit(event: str, **fields: Any):
    """Module-level shortcut for the global telemetry sink."""
    _TELEMETRY.emit(event, **fields)


@contextmanager
def trace(event: str, **fields: Any) -> Iterator[Dict[str, Any]]:
    """
    Context manager that emits a `{event}_start` and `{event}_end` pair with
    auto-generated trace_id and elapsed_ms. Mutate the yielded dict to attach
    extra fields visible at end-time only.

        with trace("rag_generate", section="VR") as t:
            ...
            t["retrieved_ids"] = [1,2,3]
    """
    trace_id = uuid.uuid4().hex[:12]
    extras: Dict[str, Any] = {}
    start = time.perf_counter()
    emit(f"{event}_start", trace_id=trace_id, **fields)
    try:
        yield extras
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Extras override fields if a key was added at end-time (e.g. updated
        # `model` from a usage dict). Merge first so we never double-pass a kw.
        merged = {**fields, **extras}
        emit(f"{event}_error",
             trace_id=trace_id, elapsed_ms=round(elapsed_ms, 1),
             error_type=type(e).__name__, error_msg=str(e)[:500],
             **merged)
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        merged = {**fields, **extras}
        emit(f"{event}_end",
             trace_id=trace_id, elapsed_ms=round(elapsed_ms, 1),
             **merged)


def aggregate(path: str = TELEMETRY_FILE, last_n: Optional[int] = None) -> Dict[str, Any]:
    """Quick summary statistics over the JSONL log — useful for dashboards."""
    if not os.path.exists(path):
        return {"events": 0}
    counts: Dict[str, int] = {}
    cost = 0.0
    in_t = out_t = cache_r = cache_w = 0
    by_section: Dict[str, int] = {}
    rows = 0
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if last_n is not None:
        lines = lines[-last_n:]
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows += 1
        counts[r.get("event", "?")] = counts.get(r.get("event", "?"), 0) + 1
        if "cost_usd" in r:
            cost += float(r["cost_usd"] or 0)
        in_t    += int(r.get("input_tokens", 0) or 0)
        out_t   += int(r.get("output_tokens", 0) or 0)
        cache_r += int(r.get("cache_read_input_tokens", 0) or 0)
        cache_w += int(r.get("cache_creation_input_tokens", 0) or 0)
        sec = r.get("section")
        if sec:
            by_section[sec] = by_section.get(sec, 0) + 1
    return {
        "events": rows, "by_event": counts, "by_section": by_section,
        "tokens": {"in": in_t, "out": out_t, "cache_read": cache_r, "cache_write": cache_w},
        "total_cost_usd": round(cost, 4),
    }
