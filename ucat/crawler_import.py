"""Import questions captured by the Crawler tool (NGMP apps/crawler) into the KB.

Reads ``<crawler_output>/manifest.json``, walks per-question ``.txt`` + ``.json``
files, groups into trainer-shape docs, and adds them to the SQLite KB with
``source='crawler'``.

Defensive about Crawler's evolving output: if its Phase 1 enrichment has landed
(adding ``stem``, ``options``, ``correctAnswer``, ``explanation`` to the per-
question JSON) we use those directly. Otherwise we parse the same fields out of
the ``.txt`` file ourselves so this works against today's crawler output.

Sections handled:
  VR / QR / SJT  — group every 4 consecutive captures with a shared passage/stimulus/scenario
  DM             — group every 5 consecutive captures (no shared context)
  AR             — skipped with a warning (UCAT removed AR)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .telemetry import emit, logger, trace

# ─── Section grouping config ──────────────────────────────────────────────────

SHARED_CONTEXT = {
    "VR":  {"count": 4, "field": "passage",  "min_len": 100},
    "QR":  {"count": 4, "field": "stimulus", "min_len": 20},
    "SJT": {"count": 4, "field": "scenario", "min_len": 50},
}

OPTION_RE = re.compile(r"^\s*([A-E])\)\s*(.+?)\s*$")

# ─── Public API ───────────────────────────────────────────────────────────────

def import_from_crawler(db, crawler_output_dir: Path) -> Dict[str, Any]:
    """
    Walk a crawler output directory and add captured questions to the KB.

    Returns a dict with:
      counts:    {VR, DM, QR, SJT}            sets imported per section
      skipped:   list of {bucket, section, reason, count}
      ar_skipped: int                          AR captures dropped
      errors:    list of strings
    """
    output = Path(crawler_output_dir)
    if not output.exists():
        raise FileNotFoundError(f"Crawler output dir not found: {output}")
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")

    with trace("crawler_import", path=str(output)) as t:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        entries = manifest.get("questions") or []
        counts = {"VR": 0, "DM": 0, "QR": 0, "SJT": 0}
        skipped: List[Dict[str, Any]] = []
        errors: List[str] = []
        ar_skipped = 0

        # Group by (bucket, section); preserve manifest order within each.
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for e in entries:
            sec = e.get("section")
            bucket = e.get("bucket") or "?"
            if sec is None:
                skipped.append({"bucket": bucket, "section": "?", "reason": "no section", "count": 1})
                continue
            if sec == "AR":
                ar_skipped += 1
                continue
            if sec not in SHARED_CONTEXT and sec != "DM":
                skipped.append({"bucket": bucket, "section": sec, "reason": "unknown section", "count": 1})
                continue
            groups.setdefault((bucket, sec), []).append(e)

        for (bucket, section), bucket_entries in groups.items():
            try:
                enriched = [_parse_entry(output, e) for e in bucket_entries]
                enriched = [x for x in enriched if x is not None]

                if section == "DM":
                    docs, leftover = _group_dm(enriched)
                else:
                    docs, leftover = _group_shared(enriched, section)

                for doc in docs:
                    db.add_doc(section, doc, source="crawler")
                    counts[section] += 1

                if leftover:
                    skipped.append({"bucket": bucket, "section": section,
                                     "reason": f"{leftover} captures dropped (incomplete chunk or short shared context)",
                                     "count": leftover})

            except Exception as e:
                logger.exception("Crawler import failed for %s/%s", bucket, section)
                errors.append(f"{bucket}/{section}: {e}")

        result = {
            "counts": counts,
            "skipped": skipped,
            "ar_skipped": ar_skipped,
            "errors": errors,
            "total": sum(counts.values()),
        }
        t.update({
            "imported_total": result["total"],
            "by_section": counts,
            "ar_skipped": ar_skipped,
            "skipped_groups": len(skipped),
            "errors": len(errors),
        })
        emit("crawler_import_summary", **result)
        return result


# ─── Per-entry parsing ────────────────────────────────────────────────────────

def _parse_entry(output_dir: Path, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read .txt and .json, return enriched dict or None on failure."""
    files = entry.get("files") or {}
    txt_rel  = files.get("txt")
    json_rel = files.get("json")
    if not txt_rel or not json_rel:
        return None

    txt_path  = output_dir / txt_rel
    json_path = output_dir / json_rel
    if not txt_path.exists() or not json_path.exists():
        return None

    try:
        text = txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = txt_path.read_text(encoding="utf-8", errors="replace")
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    # Phase 1 enrichment fields (if Crawler has them already)
    stem    = meta.get("stem")
    options = meta.get("options")
    answer  = meta.get("correctAnswer")
    expl    = meta.get("explanation")

    if not stem or not options:
        # Fall back: parse stem + options out of the raw .txt.
        parsed_stem, parsed_options = _parse_txt(text)
        stem    = stem    or parsed_stem
        options = options or parsed_options

    if not stem or not options or len(options) < 2:
        return None

    return {
        "id":      entry.get("id"),
        "section": entry.get("section"),
        "bucket":  entry.get("bucket") or "?",
        "stem":    stem.strip(),
        "options": options,
        "correctAnswer": answer,
        "explanation":   expl,
        "hasImages":     meta.get("hasImages", False),
    }


def _parse_txt(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse the on-disk .txt format:

        <passage + question stem...>

        A) option text
        B) option text
        C) option text
        ...

    The stem is everything before the first option line. Options are letter-keyed.
    """
    options: Dict[str, str] = {}
    stem_lines: List[str] = []
    seen_option = False

    # Be liberal: options can appear without a blank-line separator.
    for line in text.splitlines():
        m = OPTION_RE.match(line)
        if m:
            seen_option = True
            options[m.group(1)] = m.group(2).strip()
            continue
        if not seen_option:
            stem_lines.append(line)
        else:
            # Continuation of the previous option — append.
            if options:
                last = sorted(options.keys())[-1]
                options[last] = (options[last] + " " + line.strip()).strip()

    stem = "\n".join(stem_lines).rstrip()
    return stem, options


# ─── Grouping ─────────────────────────────────────────────────────────────────

def _longest_common_prefix(strs: List[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        i = 0
        m = min(len(prefix), len(s))
        while i < m and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def _trim_to_passage_boundary(s: str) -> str:
    """
    Trim a trailing partial question-label word out of a raw LCP.

    Medify VR sources sometimes run "...July 2024What is suggested..." so the
    LCP includes part of the next question's leading word. Find the last
    non-space lowercase→uppercase transition and cut there.
    """
    s = s.rstrip()
    for i in range(len(s) - 1, 0, -1):
        ch = s[i]
        prev = s[i - 1]
        if ch.isupper() and not prev.isupper() and not prev.isspace():
            return s[:i].rstrip()
    return s


def _detect_vr_type(options: Dict[str, str]) -> str:
    values = [v.strip().lower() for v in options.values()]
    if len(values) == 3 and set(values) == {"true", "false", "can't tell"}:
        return "tf"
    return "mc"


def _build_question(qi: int, e: Dict[str, Any], shared_prefix: str,
                     section: str) -> Dict[str, Any]:
    """Build a trainer-schema question dict from an enriched entry."""
    text = e["stem"]
    if shared_prefix:
        text = text[len(shared_prefix):].strip()
    q: Dict[str, Any] = {
        "number":  qi + 1,
        "text":    text,
        "options": e["options"],
    }
    if section == "VR":
        q["type"] = _detect_vr_type(e["options"])
    if section == "DM":
        q["type"] = "logical"  # default; refined later by enrichment
    if e.get("correctAnswer"):
        q["answer"] = e["correctAnswer"]
    if e.get("explanation"):
        q["explanation"] = e["explanation"]
    # Pre-fill placeholder difficulty + coverage so the structured schema can
    # accept these as RAG examples even before the Phase 2 enrichment lands.
    q["difficulty"] = 3.0
    q["coverage"] = {
        "topic": "imported",
        "scenario_type": "abstract",
        "contains_named_entities": False,
        "cultural_context": "UK",
    }
    return q


def _group_shared(entries: List[Dict[str, Any]], section: str
                   ) -> Tuple[List[Dict[str, Any]], int]:
    """Group VR/QR/SJT into 4-question docs sharing a passage/stimulus/scenario."""
    cfg = SHARED_CONTEXT[section]
    n_per = cfg["count"]
    field = cfg["field"]
    min_len = cfg["min_len"]

    docs: List[Dict[str, Any]] = []
    consumed = 0
    i = 0
    while i + n_per <= len(entries):
        chunk = entries[i:i + n_per]
        prefix = _trim_to_passage_boundary(_longest_common_prefix([e["stem"] for e in chunk]))
        if len(prefix) < min_len:
            i += 1
            continue
        questions = [_build_question(qi, e, prefix, section) for qi, e in enumerate(chunk)]
        doc: Dict[str, Any] = {
            "section":   section,
            "bucket":    chunk[0]["bucket"],
            "questions": questions,
        }
        doc[field] = prefix
        docs.append(doc)
        consumed += n_per
        i += n_per

    leftover = max(len(entries) - consumed, 0)
    return docs, leftover


def _group_dm(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Group DM into chunks of 5 standalone questions."""
    docs: List[Dict[str, Any]] = []
    consumed = 0
    i = 0
    while i + 5 <= len(entries):
        chunk = entries[i:i + 5]
        questions = [_build_question(qi, e, "", "DM") for qi, e in enumerate(chunk)]
        docs.append({
            "section":   "DM",
            "bucket":    chunk[0]["bucket"],
            "questions": questions,
        })
        consumed += 5
        i += 5
    leftover = max(len(entries) - consumed, 0)
    return docs, leftover
