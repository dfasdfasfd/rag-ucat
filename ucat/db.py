"""SQLite-backed knowledge base + generation history.

Tables:
  kb         — source/seed/promoted documents with embeddings
  generated  — every generated set, with usage, verdict, coverage, difficulty
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .config import DB_FILE, SECTIONS, DUPLICATE_THRESHOLD
from .retrieval import cosine_sim, mmr_select, rerank_by_similarity
from .telemetry import logger


def _doc_difficulty(data: dict) -> Optional[float]:
    """Average of per-question `difficulty` fields, if present.

    Used to seed the `kb.difficulty` column so retrieval can prefer
    examples whose difficulty matches the user's `target_difficulty`.
    Returns None if no questions have difficulty (legacy crawler-imported
    docs sometimes lack it). Clamped to the 1.0–5.0 IRT band.
    """
    qs = data.get("questions") or []
    diffs = []
    for q in qs:
        d = q.get("difficulty")
        if isinstance(d, (int, float)):
            diffs.append(float(d))
    if not diffs:
        return None
    avg = sum(diffs) / len(diffs)
    return max(1.0, min(5.0, avg))


def _content_hash(section: str, data: dict) -> str:
    """Stable content fingerprint for dedup. SHA-256 of section + canonical JSON.

    `sort_keys=True` and `separators=(",", ":")` give a deterministic byte
    sequence regardless of dict insertion order or whitespace, so two
    semantically-identical docs always hash the same. We hash `section`
    too so the same JSON under different sections doesn't collide.

    16 hex chars (64 bits) is overkill for this scale (collision probability
    after 100M docs ≈ 2.7e-7) and keeps the column compact.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{section}::{canonical}".encode("utf-8")).hexdigest()[:16]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def embed_text_for(data: dict) -> str:
    """Build a compact text representation for embedding a KB document.

    For sections with structured visuals (QR/AR/DM venn), serialize a textual
    summary so the embedding still captures the chart/shape semantics.
    """
    section = data.get("section", "")
    parts = [f"UCAT {SECTIONS.get(section, '')} question"]

    # VR — passage.
    if "passage" in data:
        parts.append(str(data["passage"])[:1200])

    # SJT — scenario (workplace/clinical situation).
    if section == "SJT" and "scenario" in data:
        parts.append(str(data["scenario"])[:1200])

    # QR — chart summary.
    if section == "QR":
        stim = data.get("stimulus")
        if isinstance(stim, dict):
            parts.append(f"Chart: {stim.get('title','')} ({stim.get('type','')})")
            parts.append(f"Categories: {', '.join(map(str, stim.get('categories', [])[:8]))}")
            for s in (stim.get("series") or [])[:4]:
                parts.append(f"Series {s.get('name','')}: {s.get('values', [])[:6]}")
        elif isinstance(stim, str):
            parts.append(stim[:600])

    # AR — rule summary.
    if section == "AR":
        parts.append(f"Set A rule: {str(data.get('set_a_rule',''))[:200]}")
        parts.append(f"Set B rule: {str(data.get('set_b_rule',''))[:200]}")
        # back-compat with legacy text descriptions
        for k in ("set_a_description", "set_b_description"):
            if data.get(k):
                parts.append(str(data[k])[:300])

    # DM — include the per-question subtype so venn/syllogism/probability sets
    # produce distinguishable embeddings. Without this, "DM question + 5
    # question texts" looks similar across all subtypes and MMR retrieves
    # mixed-subtype examples even when the user wants a venn-only context.
    if section == "DM":
        types = [q.get("type", "?") for q in (data.get("questions") or [])[:5]]
        if types:
            parts.append(f"DM types: {', '.join(types)}")

    # VR — include the per-question minigame_kind so tfc / inference / main-idea
    # / paraphrase / tone-purpose sets get distinguishable embeddings (parallel
    # to the DM types fix). Without this, two VR sets on similar topics but
    # different question structures look identical to MMR.
    if section == "VR":
        kinds = [
            q.get("minigame_kind") or q.get("type") or "?"
            for q in (data.get("questions") or [])[:5]
        ]
        if kinds:
            parts.append(f"VR types: {', '.join(kinds)}")

    # Topic + scenario tags from coverage metadata. These are the highest-
    # signal retrieval keys for "give me 4 examples on a similar topic" —
    # without them, two QR bar-charts about completely different domains
    # ("UK healthcare funding" vs "Premier League attendance") cluster in
    # embedding space because the structural text is dominant. Including
    # the topic strings gives MMR a real basis for topic diversity.
    topics: List[str] = []
    scenarios: List[str] = []
    for q in (data.get("questions") or [])[:5]:
        cov = q.get("coverage") or {}
        if isinstance(cov, dict):
            t = cov.get("topic")
            s = cov.get("scenario_type")
            if isinstance(t, str) and t:
                topics.append(t)
            if isinstance(s, str) and s:
                scenarios.append(s)
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")
    if scenarios:
        parts.append(f"Scenarios: {', '.join(scenarios)}")

    # All sections — first few question texts.
    for q in (data.get("questions") or [])[:5]:
        parts.append(q.get("text", "")[:300])

    return " ".join(p for p in parts if p)


# ─── Database ────────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path: str = DB_FILE):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS kb (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            section      TEXT NOT NULL,
            source       TEXT DEFAULT 'manual',
            data         TEXT NOT NULL,
            embed_text   TEXT NOT NULL,
            embedding    TEXT DEFAULT NULL,
            embed_model  TEXT DEFAULT NULL,
            content_hash TEXT DEFAULT NULL,
            difficulty   REAL DEFAULT NULL,
            created      TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS generated (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            section     TEXT NOT NULL,
            data        TEXT NOT NULL,
            context_ids TEXT NOT NULL,
            usage       TEXT DEFAULT NULL,
            verdict     TEXT DEFAULT NULL,
            coverage    TEXT DEFAULT NULL,
            difficulty  REAL DEFAULT NULL,
            created     TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        # Migrate older DBs that lack the new columns.
        for col, decl in (("usage", "TEXT"), ("verdict", "TEXT"),
                          ("coverage", "TEXT"), ("difficulty", "REAL")):
            try:
                c.execute(f"ALTER TABLE generated ADD COLUMN {col} {decl} DEFAULT NULL")
            except sqlite3.OperationalError:
                pass
        # Migrate kb to add content_hash if missing.
        try:
            c.execute("ALTER TABLE kb ADD COLUMN content_hash TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
        # Migrate kb to add difficulty if missing. Used for difficulty-
        # proximity scoring during retrieval — see `retrieve()`.
        try:
            c.execute("ALTER TABLE kb ADD COLUMN difficulty REAL DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
        # Migrate kb to add user_rating: explicit feedback signal for
        # retrieval weighting. NULL = no signal, +1 = upvoted (or
        # promoted), -1 = downvoted. Read by `retrieve()` to apply a
        # small score boost/penalty alongside cosine similarity.
        try:
            c.execute("ALTER TABLE kb ADD COLUMN user_rating INTEGER DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
        # Index for fast dedup lookups on import. Partial index avoids the
        # cost of indexing legacy NULL rows that haven't been backfilled yet.
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_section_hash"
            " ON kb (section, content_hash) WHERE content_hash IS NOT NULL"
        )
        # Index for fast retrieval scans (section + has-embedding filter).
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_section_embedded"
            " ON kb (section) WHERE embedding IS NOT NULL"
        )
        self.conn.commit()
        # Backfill content_hash for any pre-migration rows. Cheap one-time pass.
        self._backfill_content_hashes()
        # Same idea for difficulty — derive from per-question difficulties.
        self._backfill_difficulties()

    def _backfill_difficulties(self) -> None:
        """One-time migration: derive `difficulty` for legacy rows from
        their per-question difficulty fields. Skips silently if there's
        nothing to backfill."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id, data FROM kb WHERE difficulty IS NULL"
        ).fetchall()
        if not rows:
            return
        updates: List[Tuple[Optional[float], int]] = []
        for row_id, data_json in rows:
            try:
                data = json.loads(data_json)
            except (TypeError, ValueError):
                continue
            d = _doc_difficulty(data)
            updates.append((d, row_id))
        if updates:
            c.executemany("UPDATE kb SET difficulty=? WHERE id=?", updates)
            self.conn.commit()
            logger.info("kb migration: backfilled difficulty for %d row(s)", len(updates))

    def _backfill_content_hashes(self) -> None:
        """One-time migration: compute content_hash for legacy rows.

        Skips quietly if there's nothing to backfill. Runs on every startup —
        the WHERE filter ensures it's a no-op once the column is populated.
        """
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id, section, data FROM kb WHERE content_hash IS NULL"
        ).fetchall()
        if not rows:
            return
        updates: List[Tuple[str, int]] = []
        for row_id, section, data_json in rows:
            try:
                data = json.loads(data_json)
            except (TypeError, ValueError):
                continue  # skip un-parseable rows; they'll never dedup but won't crash
            updates.append((_content_hash(section, data), row_id))
        if updates:
            c.executemany("UPDATE kb SET content_hash=? WHERE id=?", updates)
            self.conn.commit()
            logger.info("kb migration: backfilled content_hash for %d row(s)", len(updates))

    # ── KB ──

    def add_doc(self, section: str, data: dict, source: str = "manual") -> int:
        """Insert a doc, or return the existing row's id if an exact-content
        duplicate already exists in this section. Dedup is keyed on
        SHA-256(section + canonical-JSON(data)) — see `_content_hash`.

        Re-running the enrich pipeline and re-importing the resulting
        import.json is therefore idempotent: unchanged docs re-link to
        their existing rows; only genuinely-new docs add new rows.
        """
        h = _content_hash(section, data)
        c = self.conn.cursor()
        existing = c.execute(
            "SELECT id FROM kb WHERE section=? AND content_hash=? LIMIT 1",
            (section, h),
        ).fetchone()
        if existing:
            return existing[0]
        et = embed_text_for(data)
        diff = _doc_difficulty(data)
        c.execute(
            "INSERT INTO kb (section,source,data,embed_text,content_hash,difficulty)"
            " VALUES (?,?,?,?,?,?)",
            (section, source, json.dumps(data), et, h, diff),
        )
        self.conn.commit()
        return c.lastrowid

    def doc_exists(self, section: str, data: dict) -> bool:
        """Cheap existence check by content hash, without inserting."""
        c = self.conn.cursor()
        return (
            c.execute(
                "SELECT 1 FROM kb WHERE section=? AND content_hash=? LIMIT 1",
                (section, _content_hash(section, data)),
            ).fetchone()
            is not None
        )

    def set_embedding(self, doc_id: int, vec: list, model: str):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET embedding=?,embed_model=? WHERE id=?",
                  (json.dumps(vec), model, doc_id))
        self.conn.commit()

    def set_embeddings_batch(self, items: List[Tuple[int, list, str]]):
        c = self.conn.cursor()
        c.executemany(
            "UPDATE kb SET embedding=?,embed_model=? WHERE id=?",
            [(json.dumps(v), m, i) for (i, v, m) in items],
        )
        self.conn.commit()

    def get_all_docs(self, section: Optional[str] = None, limit: int = 5000):
        c = self.conn.cursor()
        q = ("SELECT id,section,source,data,embed_text,embedding,embed_model,created FROM kb"
             + (" WHERE section=?" if section else "")
             + " ORDER BY created DESC LIMIT ?")
        args = (section, limit) if section else (limit,)
        return [
            {"id": r[0], "section": r[1], "source": r[2],
             "data": json.loads(r[3]), "embed_text": r[4],
             "embedding": json.loads(r[5]) if r[5] else None,
             "embed_model": r[6], "created": r[7]}
            for r in c.execute(q, args).fetchall()
        ]

    def get_unindexed(self, section: Optional[str] = None):
        c = self.conn.cursor()
        if section:
            rows = c.execute("SELECT id,embed_text FROM kb WHERE embedding IS NULL AND section=?", (section,)).fetchall()
        else:
            rows = c.execute("SELECT id,embed_text FROM kb WHERE embedding IS NULL").fetchall()
        return [{"id": r[0], "embed_text": r[1]} for r in rows]

    def count(self, section: Optional[str] = None, indexed_only: bool = False) -> int:
        c = self.conn.cursor()
        if section and indexed_only:
            cond, args = "WHERE section=? AND embedding IS NOT NULL", (section,)
        elif section:
            cond, args = "WHERE section=?", (section,)
        elif indexed_only:
            cond, args = "WHERE embedding IS NOT NULL", ()
        else:
            cond, args = "", ()
        return c.execute(f"SELECT COUNT(*) FROM kb {cond}", args).fetchone()[0]

    def _retrieval_index(
        self, section: str, subtype: Optional[str] = None
    ) -> List[Tuple[int, list, Optional[float], Optional[int]]]:
        """Lightweight (id, embedding-vec) tuples for indexed docs in `section`.

        Used by `retrieve` and `find_near_duplicates` to avoid loading the
        full `data` blob for every KB row on every call. We still pay
        `json.loads` per embedding vector (≈1.5K floats), but we skip
        the much larger `data` JSON until we know which rows we actually
        want to return.

        When `subtype` is provided, the index is pre-filtered via an
        ANCHORED SQL pattern match against `embed_text`. The pattern is
        section-specific because `embed_text_for` serialises subtypes
        differently per section:
          DM:  "DM types: venn, venn, ..."   → `LIKE '%DM types: %{subtype}%'`
          VR:  "VR types: tfc, tfc, ..."     → `LIKE '%VR types: %{subtype}%'`
          QR:  "Chart: <title> (bar)"        → `LIKE '%(\\{subtype}\\)%'`
          SJT: "type" lives only on questions → no LIKE pre-filter
        The earlier unanchored `LIKE '%{subtype}%'` produced false
        positives — `bar` matched a pie chart titled "Snack bar revenue",
        `probability` matched a syllogism question text mentioning the
        word, etc. Anchoring eliminates those without breaking valid
        cases.
        """
        c = self.conn.cursor()
        pattern = self._subtype_like_pattern(section, subtype) if subtype else None
        if pattern is not None:
            rows = c.execute(
                "SELECT id, embedding, difficulty, user_rating FROM kb"
                " WHERE section=? AND embedding IS NOT NULL"
                "   AND embed_text LIKE ?",
                (section, pattern),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, embedding, difficulty, user_rating FROM kb"
                " WHERE section=? AND embedding IS NOT NULL",
                (section,),
            ).fetchall()
        return [(r[0], json.loads(r[1]), r[2], r[3]) for r in rows]

    @staticmethod
    def _subtype_like_pattern(section: str, subtype: str) -> Optional[str]:
        """Section-aware LIKE pattern that matches the canonical embed_text
        serialisation produced by `embed_text_for`. Returns None if the
        section doesn't encode this subtype in embed_text (in which case
        the caller falls back to no pre-filter and relies on cosine
        ranking + the subtype-anchored query string to bias selection).
        """
        if section == "DM":
            # Match the "DM types: ..." line. The trailing `%` handles the
            # subtype appearing anywhere in the comma-separated list.
            return f"%DM types: %{subtype}%"
        if section == "VR":
            return f"%VR types: %{subtype}%"
        if section == "QR":
            # The chart-type marker is `(bar)`, `(line)`, `(stacked_bar)`, etc.
            return f"%({subtype})%"
        # SJT and AR don't currently encode subtypes in embed_text.
        return None

    def _hydrate_docs(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch full doc dicts for a small set of ids. Returned as `{id: doc}`
        so callers can re-order to match a previously-computed ranking
        without an N×M lookup."""
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id,section,source,data,embed_text,embedding,embed_model,created"
            f" FROM kb WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {
            r[0]: {
                "id": r[0], "section": r[1], "source": r[2],
                "data": json.loads(r[3]), "embed_text": r[4],
                "embedding": json.loads(r[5]) if r[5] else None,
                "embed_model": r[6], "created": r[7],
            }
            for r in rows
        }

    def retrieve(
        self,
        section: str,
        qvec: list,
        top_k: int,
        mmr_lambda: float,
        subtype: Optional[str] = None,
        target_difficulty: Optional[float] = None,
    ):
        # 1. Score every indexed doc using only (id, embedding) — cheap.
        # When subtype is set, the index is pre-filtered to docs whose
        # embed_text matches the subtype, so retrieved context stays
        # coherent with a subtype lock-in. Falls back to full-section
        # retrieval if the filtered pool is empty (e.g., user asked for
        # venn but KB has no venn-heavy docs yet).
        index = self._retrieval_index(section, subtype=subtype)
        if not index and subtype:
            logger.info(
                "retrieve: subtype filter '%s' for %s yielded 0 docs — "
                "falling back to full-section pool",
                subtype, section,
            )
            index = self._retrieval_index(section)
        if not index:
            return []
        # MMR's diversity step needs a candidate pool larger than top_k to
        # do useful selection. When the pool is exhausted (KB smaller than
        # max(top_k * 3, 8)), the function below returns all candidates
        # in cosine-similarity order — no diversity selection. Surface
        # this as a debug-grade signal so users can correlate "RAG is
        # picking the same examples every time" with "KB needs more docs".
        pool_size = max(top_k * 3, 8)
        if len(index) <= top_k:
            logger.warning(
                "retrieve: KB pool for %s%s has %d docs (top_k=%d) — "
                "MMR diversity selection is skipped; consider indexing more "
                "examples for better RAG context",
                section, f" subtype={subtype}" if subtype else "",
                len(index), top_k,
            )
        elif len(index) < pool_size:
            logger.info(
                "retrieve: KB pool for %s%s has %d docs (target pool %d) — "
                "MMR will operate on a smaller pool than ideal",
                section, f" subtype={subtype}" if subtype else "",
                len(index), pool_size,
            )
        # Compute the per-doc score:
        #   base       = cosine similarity (range -1..1, typically 0..1)
        #   difficulty = small proximity bonus when target_difficulty set
        #   rating     = small ±5% boost from explicit user feedback
        # Coefficients are tuned so cosine still dominates — these are
        # tiebreakers, not primary signals.
        DIFF_WEIGHT = 0.15
        RATING_WEIGHT = 0.05
        def _score(
            qsim: float,
            doc_diff: Optional[float],
            doc_rating: Optional[int],
        ) -> float:
            score = qsim
            if target_difficulty is not None:
                d = doc_diff if doc_diff is not None else target_difficulty
                proximity = 1.0 - abs(target_difficulty - d) / 4.0
                score += DIFF_WEIGHT * proximity
            if doc_rating:
                # +1 → +5% boost, -1 → -5% penalty.
                score += RATING_WEIGHT * doc_rating
            return score
        scored = sorted(
            (
                (_score(cosine_sim(qvec, vec), diff, rating), id_, vec)
                for id_, vec, diff, rating in index
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        # 2. Take the top-N by similarity into the MMR pool. Hydrate ONLY this
        #    pool to avoid loading the full data blob for the long tail.
        pool_slice = scored[:pool_size]
        pool_ids = [id_ for _, id_, _ in pool_slice]
        hydrated = self._hydrate_docs(pool_ids)
        # 3. Build the (sim, doc) pool and feed MMR. Skip any row that
        #    silently disappeared between the two queries (concurrent delete).
        pool = [
            (sim, hydrated[id_])
            for sim, id_, _ in pool_slice
            if id_ in hydrated
        ]
        return mmr_select(qvec, pool, top_k, mmr_lambda)

    def find_near_duplicates(self, section: str, qvec: list,
                              threshold: float = DUPLICATE_THRESHOLD):
        """Return [(sim, doc), ...] for KB items whose similarity ≥ threshold.
        Same lightweight-index optimization as `retrieve`: we score every
        indexed doc cheaply, then hydrate only the ones over threshold."""
        index = self._retrieval_index(section)
        if not index:
            return []
        hits = [
            (sim, id_)
            for id_, vec, _diff, _rating in index
            for sim in (cosine_sim(qvec, vec),)
            if sim >= threshold
        ]
        if not hits:
            return []
        hydrated = self._hydrate_docs([id_ for _, id_ in hits])
        return [(sim, hydrated[id_]) for sim, id_ in hits if id_ in hydrated]

    def import_json(self, path: str) -> Dict[str, int]:
        """Import a JSON file containing a list of section docs.

        Returns counts so the caller can tell new imports from re-imports of
        already-known docs:

          {
            "added":   <int>,  # new rows inserted into the KB
            "skipped": <int>,  # exact-content duplicates (re-runs of enrich)
            "ignored": <int>,  # malformed entries (bad shape, unknown
                               # section, or fails Pydantic validation)
            "total":   <int>,  # everything in the input file
          }

        Each item is validated against its section's Pydantic model
        (`SECTION_MODELS`) before insertion. This catches structurally
        invalid items — empty `questions`, missing `passage`, wrong
        question count, etc. — that would otherwise pollute the
        retrieval pool with embedding-text-only-section-name garbage.

        Dedup is based on a SHA-256 of (section + canonical-JSON(data)) — see
        `add_doc`. Re-importing the same `import.json` after re-running enrich
        is idempotent: every doc returns its existing row id and `skipped`
        equals the file length.
        """
        # Imported here to avoid a top-level circular import (models →
        # config → db is a known shape; db → models would close the loop).
        from .models import SECTION_MODELS
        from pydantic import ValidationError

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else [raw]
        added = 0
        skipped = 0
        ignored = 0
        for item in items:
            if not (isinstance(item, dict) and item.get("section") in SECTIONS):
                ignored += 1
                continue
            section = item["section"]
            # Pydantic validation: rejects malformed docs (empty questions,
            # wrong count, missing required fields) before they pollute the KB.
            model_cls = SECTION_MODELS.get(section)
            if model_cls is None:
                ignored += 1
                continue
            try:
                model_cls.model_validate(item)
            except ValidationError as e:
                logger.warning(
                    "import_json: dropped malformed %s doc: %s",
                    section, str(e)[:200],
                )
                ignored += 1
                continue
            if self.doc_exists(section, item):
                skipped += 1
                continue
            self.add_doc(section, item, source="imported")
            added += 1
        return {
            "added":   added,
            "skipped": skipped,
            "ignored": ignored,
            "total":   len(items),
        }

    # ── Generated ──

    def add_generated(
        self,
        section: str,
        data: dict,
        ctx_ids: list,
        *,
        usage: Optional[dict] = None,
        verdict: Optional[dict] = None,
        coverage: Optional[dict] = None,
        difficulty: Optional[float] = None,
    ) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO generated (section,data,context_ids,usage,verdict,coverage,difficulty)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                section, json.dumps(data), json.dumps(ctx_ids),
                json.dumps(usage)    if usage    else None,
                json.dumps(verdict)  if verdict  else None,
                json.dumps(coverage) if coverage else None,
                difficulty,
            ),
        )
        self.conn.commit()
        return c.lastrowid

    def update_generated_verdict(
        self,
        row_id: int,
        *,
        verdict: Optional[dict] = None,
        usage: Optional[dict] = None,
        difficulty: Optional[float] = None,
    ) -> None:
        """Patch a previously-inserted row with the deferred verify outcome.

        Async verify completes after add_generated has returned, so we need to
        rewrite verdict/usage/difficulty in-place (rather than insert a duplicate
        row). All three are written together so callers don't have to juggle
        partial updates.
        """
        c = self.conn.cursor()
        c.execute(
            "UPDATE generated SET verdict=?, usage=?, difficulty=? WHERE id=?",
            (
                json.dumps(verdict)  if verdict  else None,
                json.dumps(usage)    if usage    else None,
                difficulty,
                row_id,
            ),
        )
        self.conn.commit()

    def get_generated(self, limit: int = 500):
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id,section,data,context_ids,usage,verdict,coverage,difficulty,created"
            " FROM generated ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{
            "id": r[0], "section": r[1], "data": json.loads(r[2]),
            "context_ids": json.loads(r[3]),
            "usage":    json.loads(r[4]) if r[4] else None,
            "verdict":  json.loads(r[5]) if r[5] else None,
            "coverage": json.loads(r[6]) if r[6] else None,
            "difficulty": r[7],
            "created": r[8],
        } for r in rows]

    def promote_to_kb(
        self,
        gen_id: int,
        *,
        embed_fn: Optional[Callable[[str, str], list]] = None,
        embed_model: Optional[str] = None,
    ) -> bool:
        """Promote a generated row to the KB.

        Promote IS the trainer's current implicit positive-feedback
        signal — the new KB row is created with `user_rating=1`, which
        `retrieve()` uses as a small score boost against unrated docs.
        Explicit downvoting via `set_user_rating(doc_id, -1)` is
        available if you want a row in retrieval but down-weighted
        (e.g., a doc that's an example of what NOT to generate).

        If `embed_fn` and `embed_model` are provided, the new row gets its
        embedding computed and persisted in the same call — no manual
        re-index step needed before the row participates in retrieval.
        Without those, the row is added with `embedding=NULL` (legacy
        behaviour, still works after a manual ⊛ Index Knowledge Base run).

        `embed_fn` should match the signature `embed_fn(text, model) ->
        List[float]` (i.e., `llm.embed_doc`). Passed in by the caller to
        avoid pulling the LLM module into the db layer's import graph.
        """
        c = self.conn.cursor()
        row = c.execute("SELECT section,data FROM generated WHERE id=?", (gen_id,)).fetchone()
        if not row:
            return False
        data = json.loads(row[1])
        new_id = self.add_doc(row[0], data, source="generated")
        # Mark as upvoted: promote = implicit positive feedback.
        c.execute("UPDATE kb SET user_rating=? WHERE id=?", (1, new_id))
        self.conn.commit()
        if embed_fn and embed_model:
            try:
                vec = embed_fn(embed_text_for(data), embed_model)
                self.set_embedding(new_id, vec, embed_model)
            except Exception as e:
                # Embedding is a nice-to-have on promote; if it fails the
                # row is still in the KB and a manual re-index will pick
                # it up. Don't fail the promote.
                logger.warning(
                    "promote_to_kb: row %s added but embed failed: %s",
                    new_id, e,
                )
        return True

    def set_user_rating(self, doc_id: int, rating: Optional[int]) -> None:
        """Set explicit user feedback on a KB row. ``rating`` is one of
        +1 (upvote), -1 (downvote), or None (clear).

        Used by `retrieve()` to weight cosine similarity. Rating boosts
        are small (max ±5%) — the goal is to nudge ranking, not override
        relevance. A consistently-bad doc that the user wants to keep in
        the KB for reference (e.g. as an "avoid this style" example)
        can be downvoted to push it out of the retrieval pool without
        deletion.
        """
        if rating not in (None, -1, 0, 1):
            raise ValueError(f"rating must be -1, 0, +1, or None (got {rating!r})")
        c = self.conn.cursor()
        c.execute("UPDATE kb SET user_rating=? WHERE id=?", (rating, doc_id))
        self.conn.commit()

    def coverage_stats(self, section: Optional[str] = None,
                        last_n: int = 200) -> Dict[str, Any]:
        """Distribution of topics/scenario types across recent generations.
        Used by the bias/coverage dashboard."""
        rows = self.get_generated(limit=last_n)
        if section:
            rows = [r for r in rows if r["section"] == section]

        topics: Dict[str, int] = {}
        scenarios: Dict[str, int] = {}
        sjt_situations: Dict[str, int] = {}
        diffs: List[float] = []
        for r in rows:
            cov = r.get("coverage") or {}
            for tag in (cov.get("per_question") or []):
                t = tag.get("topic", "?")
                s = tag.get("scenario_type", "?")
                topics[t] = topics.get(t, 0) + 1
                scenarios[s] = scenarios.get(s, 0) + 1
            sit = cov.get("sjt_situation_type")
            if sit:
                sjt_situations[sit] = sjt_situations.get(sit, 0) + 1
            if r.get("difficulty") is not None:
                diffs.append(float(r["difficulty"]))

        return {
            "rows": len(rows),
            "topics": topics,
            "scenarios": scenarios,
            "sjt_situations": sjt_situations,
            "difficulty": {
                "mean": (sum(diffs) / len(diffs)) if diffs else None,
                "min":  min(diffs) if diffs else None,
                "max":  max(diffs) if diffs else None,
                "count": len(diffs),
            },
        }

    def close(self):
        try:
            self.conn.close()
        except Exception as e:
            logger.warning("DB close: %s", e)
