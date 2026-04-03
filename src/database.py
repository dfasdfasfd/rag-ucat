"""
Database layer for UCAT Trainer.
SQLite storage with optional FTS5 BM25, schema migration, feedback tracking,
calibration state, and prompt versioning.
"""

import sqlite3
import json
from datetime import datetime

from src.config import DB_FILE, SECTIONS


# ─── FTS5 Capability Probe ───────────────────────────────────────────────────

def _has_fts5() -> bool:
    """Check if SQLite was compiled with FTS5 support."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.close()
        return True
    except Exception:
        return False


HAS_FTS5 = _has_fts5()


class Database:
    def __init__(self, path: str = DB_FILE):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.has_fts5 = HAS_FTS5
        self._init()
        self._migrate()

    # ─── Schema Init ─────────────────────────────────────────────────────────

    def _init(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS kb (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            section         TEXT NOT NULL,
            source          TEXT DEFAULT 'manual',
            data            TEXT NOT NULL,
            embed_text      TEXT NOT NULL,
            embedding       TEXT DEFAULT NULL,
            embed_model     TEXT DEFAULT NULL,
            created         TEXT DEFAULT CURRENT_TIMESTAMP,
            quality_score   REAL DEFAULT NULL,
            generation_count INTEGER DEFAULT 0,
            success_count   INTEGER DEFAULT 0,
            implicit_score  REAL DEFAULT 0.0,
            difficulty_est  REAL DEFAULT NULL,
            topic_cluster   TEXT DEFAULT NULL,
            data_type       TEXT DEFAULT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS generated (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            section         TEXT NOT NULL,
            data            TEXT NOT NULL,
            context_ids     TEXT NOT NULL,
            created         TEXT DEFAULT CURRENT_TIMESTAMP,
            view_duration_ms INTEGER DEFAULT 0,
            was_regenerated INTEGER DEFAULT 0,
            was_exported    INTEGER DEFAULT 0,
            prompt_version  TEXT DEFAULT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS quality_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            gen_id          INTEGER,
            rule_score      REAL,
            llm_score       REAL,
            final_score     REAL,
            dedup_kb_sim    REAL,
            dedup_session_sim REAL,
            format_valid    INTEGER,
            errors          TEXT,
            auto_promoted   INTEGER DEFAULT 0,
            created         TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (gen_id) REFERENCES generated(id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS calibration_state (
            section             TEXT PRIMARY KEY,
            consecutive_approvals INTEGER DEFAULT 0,
            is_calibrated       INTEGER DEFAULT 0,
            locked_params       TEXT DEFAULT NULL,
            calibrated_at       TEXT DEFAULT NULL
        )""")
        # Initialize calibration rows for all sections
        for sec in SECTIONS:
            c.execute(
                "INSERT OR IGNORE INTO calibration_state (section) VALUES (?)",
                (sec,)
            )
        self.conn.commit()

        # FTS5 virtual table for BM25 hybrid search
        if self.has_fts5:
            try:
                c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts
                    USING fts5(embed_text, content=kb, content_rowid=id)""")
                self.conn.commit()
            except Exception:
                self.has_fts5 = False

    def _migrate(self):
        """Add columns that may be missing from older databases."""
        c = self.conn.cursor()
        existing_kb_cols = {row[1] for row in c.execute("PRAGMA table_info(kb)").fetchall()}
        existing_gen_cols = {row[1] for row in c.execute("PRAGMA table_info(generated)").fetchall()}

        kb_migrations = {
            "quality_score": "REAL DEFAULT NULL",
            "generation_count": "INTEGER DEFAULT 0",
            "success_count": "INTEGER DEFAULT 0",
            "implicit_score": "REAL DEFAULT 0.0",
            "difficulty_est": "REAL DEFAULT NULL",
            "topic_cluster": "TEXT DEFAULT NULL",
            "data_type": "TEXT DEFAULT NULL",
        }
        gen_migrations = {
            "view_duration_ms": "INTEGER DEFAULT 0",
            "was_regenerated": "INTEGER DEFAULT 0",
            "was_exported": "INTEGER DEFAULT 0",
            "prompt_version": "TEXT DEFAULT NULL",
        }

        for col, typedef in kb_migrations.items():
            if col not in existing_kb_cols:
                c.execute(f"ALTER TABLE kb ADD COLUMN {col} {typedef}")

        for col, typedef in gen_migrations.items():
            if col not in existing_gen_cols:
                c.execute(f"ALTER TABLE generated ADD COLUMN {col} {typedef}")

        self.conn.commit()

    # ─── FTS5 Sync ───────────────────────────────────────────────────────────

    def _sync_fts(self, doc_id: int, embed_text: str):
        """Insert or update the FTS5 index for a single document."""
        if not self.has_fts5:
            return
        c = self.conn.cursor()
        try:
            c.execute("INSERT OR REPLACE INTO kb_fts(rowid, embed_text) VALUES (?, ?)",
                      (doc_id, embed_text))
            self.conn.commit()
        except Exception:
            pass

    def rebuild_fts(self):
        """Rebuild the entire FTS5 index from kb table."""
        if not self.has_fts5:
            return
        c = self.conn.cursor()
        c.execute("DELETE FROM kb_fts")
        rows = c.execute("SELECT id, embed_text FROM kb").fetchall()
        for doc_id, embed_text in rows:
            c.execute("INSERT INTO kb_fts(rowid, embed_text) VALUES (?, ?)",
                      (doc_id, embed_text))
        self.conn.commit()

    # ─── KB Operations ───────────────────────────────────────────────────────

    def add_doc(self, section: str, data: dict, embed_text: str,
                source: str = "manual", data_type: str = None) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO kb (section, source, data, embed_text, data_type) VALUES (?,?,?,?,?)",
            (section, source, json.dumps(data), embed_text, data_type)
        )
        self.conn.commit()
        doc_id = c.lastrowid
        self._sync_fts(doc_id, embed_text)
        return doc_id

    def set_embedding(self, doc_id: int, vec: list, model: str):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET embedding=?, embed_model=? WHERE id=?",
                  (json.dumps(vec), model, doc_id))
        self.conn.commit()

    def update_embed_text(self, doc_id: int, embed_text: str):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET embed_text=? WHERE id=?", (embed_text, doc_id))
        self.conn.commit()
        self._sync_fts(doc_id, embed_text)

    def get_all_docs(self, section=None, limit=5000):
        c = self.conn.cursor()
        q = ("SELECT id, section, source, data, embed_text, embedding, embed_model, "
             "created, quality_score, generation_count, success_count, implicit_score, "
             "difficulty_est, topic_cluster, data_type FROM kb"
             + (" WHERE section=?" if section else "")
             + " ORDER BY created DESC LIMIT ?")
        args = (section, limit) if section else (limit,)
        return [
            {
                "id": r[0], "section": r[1], "source": r[2],
                "data": json.loads(r[3]), "embed_text": r[4],
                "embedding": json.loads(r[5]) if r[5] else None,
                "embed_model": r[6], "created": r[7],
                "quality_score": r[8], "generation_count": r[9],
                "success_count": r[10], "implicit_score": r[11],
                "difficulty_est": r[12], "topic_cluster": r[13],
                "data_type": r[14],
            }
            for r in c.execute(q, args).fetchall()
        ]

    def get_unindexed(self, section=None):
        c = self.conn.cursor()
        if section:
            rows = c.execute(
                "SELECT id, embed_text FROM kb WHERE embedding IS NULL AND section=?",
                (section,)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, embed_text FROM kb WHERE embedding IS NULL"
            ).fetchall()
        return [{"id": r[0], "embed_text": r[1]} for r in rows]

    def get_model_mismatched(self, target_model: str, section=None):
        """Return docs whose embed_model differs from target (need re-embedding)."""
        c = self.conn.cursor()
        if section:
            rows = c.execute(
                "SELECT id, embed_text FROM kb WHERE embed_model != ? AND section=? AND embedding IS NOT NULL",
                (target_model, section)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, embed_text FROM kb WHERE embed_model != ? AND embedding IS NOT NULL",
                (target_model,)
            ).fetchall()
        return [{"id": r[0], "embed_text": r[1]} for r in rows]

    def count(self, section=None, indexed_only=False):
        c = self.conn.cursor()
        cond_parts = []
        args = []
        if section:
            cond_parts.append("section=?")
            args.append(section)
        if indexed_only:
            cond_parts.append("embedding IS NOT NULL")
        cond = " WHERE " + " AND ".join(cond_parts) if cond_parts else ""
        return c.execute(f"SELECT COUNT(*) FROM kb{cond}", args).fetchone()[0]

    def delete_doc(self, doc_id: int):
        c = self.conn.cursor()
        c.execute("DELETE FROM kb WHERE id=?", (doc_id,))
        if self.has_fts5:
            try:
                c.execute("DELETE FROM kb_fts WHERE rowid=?", (doc_id,))
            except Exception:
                pass
        self.conn.commit()

    def import_json(self, path: str, embed_text_fn=None) -> int:
        """Import questions from a JSON file. embed_text_fn builds embed text from data."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else [raw]
        n = 0
        for item in items:
            if isinstance(item, dict) and item.get("section") in SECTIONS:
                et = embed_text_fn(item, item["section"]) if embed_text_fn else json.dumps(item)[:500]
                self.add_doc(item["section"], item, et, source="imported")
                n += 1
        return n

    # ─── BM25 Search (FTS5) ──────────────────────────────────────────────────

    def fts5_search(self, query: str, section: str = None, limit: int = 20):
        """BM25 keyword search via FTS5. Returns [(doc_id, rank), ...]."""
        if not self.has_fts5:
            return []
        c = self.conn.cursor()
        try:
            if section:
                rows = c.execute("""
                    SELECT kb_fts.rowid, rank FROM kb_fts
                    JOIN kb ON kb.id = kb_fts.rowid
                    WHERE kb_fts MATCH ? AND kb.section = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, section, limit)).fetchall()
            else:
                rows = c.execute("""
                    SELECT rowid, rank FROM kb_fts
                    WHERE kb_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit)).fetchall()
            return [(r[0], r[1]) for r in rows]
        except Exception:
            return []

    # ─── Feedback Tracking ───────────────────────────────────────────────────

    def increment_generation_count(self, doc_ids: list):
        c = self.conn.cursor()
        for doc_id in doc_ids:
            c.execute("UPDATE kb SET generation_count = generation_count + 1 WHERE id=?",
                      (doc_id,))
        self.conn.commit()

    def increment_success_count(self, doc_ids: list):
        c = self.conn.cursor()
        for doc_id in doc_ids:
            c.execute("UPDATE kb SET success_count = success_count + 1 WHERE id=?",
                      (doc_id,))
        self.conn.commit()

    def get_implicit_score(self, doc_id: int) -> float:
        c = self.conn.cursor()
        row = c.execute("SELECT implicit_score FROM kb WHERE id=?", (doc_id,)).fetchone()
        return row[0] if row else 0.0

    def set_implicit_score(self, doc_id: int, score: float):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET implicit_score=? WHERE id=?", (score, doc_id))
        self.conn.commit()

    def set_topic_cluster(self, doc_id: int, cluster: str):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET topic_cluster=? WHERE id=?", (cluster, doc_id))
        self.conn.commit()

    def set_data_type(self, doc_id: int, data_type: str):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET data_type=? WHERE id=?", (data_type, doc_id))
        self.conn.commit()

    def set_difficulty_est(self, doc_id: int, difficulty: float):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET difficulty_est=? WHERE id=?", (difficulty, doc_id))
        self.conn.commit()

    # ─── Generated Operations ────────────────────────────────────────────────

    def add_generated(self, section: str, data: dict, ctx_ids: list,
                      prompt_version: str = None) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO generated (section, data, context_ids, prompt_version) VALUES (?,?,?,?)",
            (section, json.dumps(data), json.dumps(ctx_ids), prompt_version)
        )
        self.conn.commit()
        return c.lastrowid

    def get_generated(self, limit: int = 500):
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id, section, data, context_ids, created, view_duration_ms, "
            "was_regenerated, was_exported, prompt_version "
            "FROM generated ORDER BY created DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [
            {
                "id": r[0], "section": r[1], "data": json.loads(r[2]),
                "context_ids": json.loads(r[3]), "created": r[4],
                "view_duration_ms": r[5], "was_regenerated": r[6],
                "was_exported": r[7], "prompt_version": r[8],
            }
            for r in rows
        ]

    def update_generated_signal(self, gen_id: int, field: str, value):
        """Update an implicit feedback signal on a generated record."""
        allowed = {"view_duration_ms", "was_regenerated", "was_exported"}
        if field not in allowed:
            return
        c = self.conn.cursor()
        c.execute(f"UPDATE generated SET {field}=? WHERE id=?", (value, gen_id))
        self.conn.commit()

    def promote_to_kb(self, gen_id: int, embed_text_fn=None) -> bool:
        """Promote a generated question set to the KB."""
        c = self.conn.cursor()
        row = c.execute("SELECT section, data FROM generated WHERE id=?", (gen_id,)).fetchone()
        if row:
            data = json.loads(row[1])
            et = embed_text_fn(data, row[0]) if embed_text_fn else json.dumps(data)[:500]
            self.add_doc(row[0], data, et, source="generated")
            return True
        return False

    def promote_data_to_kb(self, section: str, data: dict, embed_text: str,
                           source: str = "batch") -> int:
        """Promote raw data directly to KB (used by batch generator)."""
        return self.add_doc(section, data, embed_text, source=source)

    # ─── Quality Log ─────────────────────────────────────────────────────────

    def add_quality_log(self, gen_id: int, rule_score: float, llm_score: float,
                        final_score: float, dedup_kb_sim: float,
                        dedup_session_sim: float, format_valid: bool,
                        errors: list, auto_promoted: bool = False):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO quality_log (gen_id, rule_score, llm_score, final_score, "
            "dedup_kb_sim, dedup_session_sim, format_valid, errors, auto_promoted) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (gen_id, rule_score, llm_score, final_score, dedup_kb_sim,
             dedup_session_sim, 1 if format_valid else 0,
             json.dumps(errors), 1 if auto_promoted else 0)
        )
        self.conn.commit()

    def get_quality_for_gen(self, gen_id: int) -> dict:
        """Get quality scores for a specific generated question set."""
        c = self.conn.cursor()
        row = c.execute(
            "SELECT rule_score, llm_score, final_score FROM quality_log "
            "WHERE gen_id=? ORDER BY created DESC LIMIT 1",
            (gen_id,)
        ).fetchone()
        if row:
            return {"rule_score": row[0], "llm_score": row[1], "final_score": row[2]}
        return None

    def get_quality_scores_batch(self, gen_ids: list) -> dict:
        """Get quality scores for multiple gen_ids at once. Returns {gen_id: {scores}}."""
        if not gen_ids:
            return {}
        c = self.conn.cursor()
        placeholders = ",".join("?" for _ in gen_ids)
        rows = c.execute(f"""
            SELECT gen_id, rule_score, llm_score, final_score
            FROM quality_log
            WHERE gen_id IN ({placeholders})
        """, gen_ids).fetchall()
        result = {}
        for r in rows:
            result[r[0]] = {"rule_score": r[1], "llm_score": r[2], "final_score": r[3]}
        return result

    def get_quality_by_prompt_version(self, prompt_version: str):
        """Get average quality scores for a specific prompt version (A/B tracking)."""
        c = self.conn.cursor()
        rows = c.execute("""
            SELECT AVG(ql.rule_score), AVG(ql.llm_score), AVG(ql.final_score), COUNT(*)
            FROM quality_log ql
            JOIN generated g ON g.id = ql.gen_id
            WHERE g.prompt_version = ?
        """, (prompt_version,)).fetchone()
        return {
            "avg_rule": rows[0], "avg_llm": rows[1],
            "avg_final": rows[2], "count": rows[3],
        }

    def get_all_prompt_versions(self) -> list:
        """Get all distinct prompt versions with their average quality scores."""
        c = self.conn.cursor()
        rows = c.execute("""
            SELECT g.prompt_version,
                   AVG(ql.rule_score), AVG(ql.llm_score), AVG(ql.final_score),
                   COUNT(*), MIN(g.created), MAX(g.created)
            FROM quality_log ql
            JOIN generated g ON g.id = ql.gen_id
            WHERE g.prompt_version IS NOT NULL
            GROUP BY g.prompt_version
            ORDER BY MAX(g.created) DESC
        """).fetchall()
        return [
            {
                "version": r[0], "avg_rule": r[1], "avg_llm": r[2],
                "avg_final": r[3], "count": r[4],
                "first_used": r[5], "last_used": r[6],
            }
            for r in rows
        ]

    # ─── Calibration State ───────────────────────────────────────────────────

    def get_calibration_state(self, section: str) -> dict:
        c = self.conn.cursor()
        row = c.execute(
            "SELECT consecutive_approvals, is_calibrated, locked_params, calibrated_at "
            "FROM calibration_state WHERE section=?",
            (section,)
        ).fetchone()
        if row:
            return {
                "consecutive_approvals": row[0],
                "is_calibrated": bool(row[1]),
                "locked_params": json.loads(row[2]) if row[2] else None,
                "calibrated_at": row[3],
            }
        return {"consecutive_approvals": 0, "is_calibrated": False,
                "locked_params": None, "calibrated_at": None}

    def update_calibration(self, section: str, consecutive_approvals: int,
                           is_calibrated: bool = False, locked_params: dict = None):
        c = self.conn.cursor()
        cal_at = datetime.now().isoformat() if is_calibrated else None
        c.execute(
            "UPDATE calibration_state SET consecutive_approvals=?, is_calibrated=?, "
            "locked_params=?, calibrated_at=? WHERE section=?",
            (consecutive_approvals, 1 if is_calibrated else 0,
             json.dumps(locked_params) if locked_params else None,
             cal_at, section)
        )
        self.conn.commit()

    # ─── Cleanup ─────────────────────────────────────────────────────────────

    def close(self):
        self.conn.close()
