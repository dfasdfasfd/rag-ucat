"""Tests for the SQLite KB layer.

Covers behaviors added by the dedup + retrieval-perf refactor:

  - `add_doc` returns the existing row id on content-hash collision (no dup row inserted)
  - `import_json` reports added / skipped / ignored counts and is idempotent
  - `_content_hash` is order-independent (same data → same hash)
  - `retrieve` uses the lightweight index and returns hydrated docs
  - Backfill migration populates content_hash for legacy rows
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# Make the parent dir importable so `import ucat...` works when running pytest from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ucat.db import Database, _content_hash, embed_text_for  # noqa: E402


@pytest.fixture
def tmp_db():
    """Throwaway SQLite DB for each test."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    db = Database(path=path)
    yield db
    db.close()
    os.unlink(path)


def _vr_doc(passage_seed: str = "P", q1: str = "Question 1?") -> dict:
    return {
        "section": "VR",
        "passage": f"{passage_seed}. " + "x" * 100,
        "questions": [
            {"number": i, "text": q1 if i == 1 else f"Question {i}?",
             "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
             "answer": "A", "explanation": "x",
             "difficulty": 3.0,
             "coverage": {"topic": "test", "scenario_type": "everyday",
                          "contains_named_entities": False}}
            for i in range(1, 5)
        ],
    }


# ─── _content_hash ────────────────────────────────────────────────────────────

class TestContentHash:
    def test_same_data_yields_same_hash(self):
        d = _vr_doc()
        assert _content_hash("VR", d) == _content_hash("VR", d)

    def test_dict_key_order_does_not_change_hash(self):
        # Two dicts with same content but different insertion order
        a = {"section": "VR", "passage": "p", "questions": []}
        b = {"questions": [], "passage": "p", "section": "VR"}
        assert _content_hash("VR", a) == _content_hash("VR", b)

    def test_different_section_yields_different_hash(self):
        d = _vr_doc()
        # Same dict body, different section namespace
        assert _content_hash("VR", d) != _content_hash("DM", d)

    def test_content_change_yields_different_hash(self):
        a = _vr_doc(passage_seed="A")
        b = _vr_doc(passage_seed="B")
        assert _content_hash("VR", a) != _content_hash("VR", b)


# ─── add_doc dedup ────────────────────────────────────────────────────────────

class TestAddDocDedup:
    def test_first_insert_creates_row(self, tmp_db: Database):
        d = _vr_doc()
        rid = tmp_db.add_doc("VR", d)
        assert rid > 0

    def test_duplicate_returns_existing_id_without_inserting_new_row(self, tmp_db: Database):
        d = _vr_doc()
        rid1 = tmp_db.add_doc("VR", d)
        rid2 = tmp_db.add_doc("VR", d, source="something-else")
        assert rid2 == rid1
        assert tmp_db.count(section="VR") == 1

    def test_different_data_creates_separate_rows(self, tmp_db: Database):
        a = _vr_doc(passage_seed="A")
        b = _vr_doc(passage_seed="B")
        rid_a = tmp_db.add_doc("VR", a)
        rid_b = tmp_db.add_doc("VR", b)
        assert rid_a != rid_b
        assert tmp_db.count(section="VR") == 2

    def test_doc_exists_matches_add_doc(self, tmp_db: Database):
        d = _vr_doc()
        assert not tmp_db.doc_exists("VR", d)
        tmp_db.add_doc("VR", d)
        assert tmp_db.doc_exists("VR", d)


# ─── import_json idempotency ──────────────────────────────────────────────────

class TestImportJsonIdempotency:
    def test_import_reports_added_skipped_ignored(self, tmp_db: Database, tmp_path):
        path = tmp_path / "import.json"
        items = [
            _vr_doc(passage_seed="A"),
            _vr_doc(passage_seed="B"),
            {"section": "VR"},                                  # malformed: no passage/questions
            {"section": "BOGUS"},                               # unknown section
            "not a dict",                                        # malformed
        ]
        path.write_text(json.dumps(items))
        result = tmp_db.import_json(str(path))
        # 2 well-formed VR docs added.
        # 3 items ignored: bogus-section + non-dict + the structurally-invalid
        # `{"section": "VR"}` (Pydantic catches the missing passage/questions).
        assert result["added"] == 2
        assert result["skipped"] == 0
        assert result["ignored"] == 3
        assert result["total"] == 5

    def test_import_rejects_pydantic_invalid_docs(self, tmp_db: Database, tmp_path):
        """Specifically guard the schema-validation path: a doc that has
        the right section + key shape but fails Pydantic constraints should
        be ignored, not silently added."""
        path = tmp_path / "import.json"
        items = [
            # Wrong question count (3 instead of 4)
            {
                "section": "VR",
                "passage": "P. " + "x" * 100,
                "questions": [
                    {"number": i, "text": f"Q{i}",
                     "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                     "answer": "A", "explanation": "x", "difficulty": 3.0,
                     "coverage": {"topic": "t", "scenario_type": "everyday",
                                  "contains_named_entities": False}}
                    for i in range(1, 4)  # only 3 — VR requires exactly 4
                ],
            },
        ]
        path.write_text(json.dumps(items))
        result = tmp_db.import_json(str(path))
        assert result["added"] == 0
        assert result["ignored"] == 1
        assert tmp_db.count(section="VR") == 0

    def test_reimporting_same_file_skips_all(self, tmp_db: Database, tmp_path):
        path = tmp_path / "import.json"
        path.write_text(json.dumps([_vr_doc(passage_seed="A"), _vr_doc(passage_seed="B")]))

        first = tmp_db.import_json(str(path))
        assert first["added"] == 2 and first["skipped"] == 0

        second = tmp_db.import_json(str(path))
        assert second["added"] == 0
        assert second["skipped"] == 2
        assert tmp_db.count(section="VR") == 2  # no duplicate rows

    def test_partial_overlap_only_adds_new(self, tmp_db: Database, tmp_path):
        first_path = tmp_path / "first.json"
        first_path.write_text(json.dumps([_vr_doc(passage_seed="A")]))
        tmp_db.import_json(str(first_path))

        second_path = tmp_path / "second.json"
        second_path.write_text(
            json.dumps([_vr_doc(passage_seed="A"), _vr_doc(passage_seed="B")])
        )
        result = tmp_db.import_json(str(second_path))
        assert result["added"] == 1
        assert result["skipped"] == 1


# ─── retrieve ─────────────────────────────────────────────────────────────────

class TestRetrieve:
    def _seed_with_embeddings(self, db: Database) -> list[int]:
        ids = []
        for seed in ("apple", "banana", "carrot"):
            ids.append(db.add_doc("VR", _vr_doc(passage_seed=seed)))
        # Manually attach embeddings — simulate the indexer having run
        # Vectors are tiny (3-D) but cosine sim still works
        db.set_embedding(ids[0], [1.0, 0.0, 0.0], "test-model")
        db.set_embedding(ids[1], [0.0, 1.0, 0.0], "test-model")
        db.set_embedding(ids[2], [0.0, 0.0, 1.0], "test-model")
        return ids

    def test_returns_empty_when_no_docs(self, tmp_db: Database):
        assert tmp_db.retrieve("VR", [1.0, 0.0, 0.0], top_k=4, mmr_lambda=0.5) == []

    def test_returns_top_k_ranked_by_similarity(self, tmp_db: Database):
        ids = self._seed_with_embeddings(tmp_db)
        results = tmp_db.retrieve("VR", [1.0, 0.0, 0.0], top_k=2, mmr_lambda=1.0)
        # lambda=1.0 means pure relevance ranking — first result must be the [1,0,0] doc
        assert len(results) <= 2
        first_score, first_doc = results[0]
        assert first_doc["id"] == ids[0]
        assert first_score == pytest.approx(1.0, abs=0.001)
        # Returned docs must be hydrated (have data + embedding fields)
        assert "data" in first_doc
        assert "embedding" in first_doc

    def test_skips_docs_without_embeddings(self, tmp_db: Database):
        # Add a doc but don't index it
        tmp_db.add_doc("VR", _vr_doc(passage_seed="unindexed"))
        ids = self._seed_with_embeddings(tmp_db)
        results = tmp_db.retrieve("VR", [1.0, 0.0, 0.0], top_k=10, mmr_lambda=1.0)
        # Should only see the 3 indexed docs, not the unindexed one
        assert len(results) == 3
        assert all(d["id"] in ids for _, d in results)

    def test_target_difficulty_breaks_ties_in_retrieval(self, tmp_db: Database):
        """When two docs have similar cosine similarity, the one whose
        difficulty is closer to `target_difficulty` should rank higher.
        Coefficient is 0.15 max, so cosine still dominates — this only
        matters as a soft tiebreaker."""
        # Create two docs with identical embeddings but different difficulties.
        from ucat.db import _doc_difficulty  # noqa
        easy_doc = _vr_doc(passage_seed="easy")
        hard_doc = _vr_doc(passage_seed="hard")
        # Override per-question difficulty so kb.difficulty lands at 1.5 vs 4.5.
        for q in easy_doc["questions"]:
            q["difficulty"] = 1.5
        for q in hard_doc["questions"]:
            q["difficulty"] = 4.5

        easy_id = tmp_db.add_doc("VR", easy_doc)
        hard_id = tmp_db.add_doc("VR", hard_doc)
        # Both get the same embedding so cosine sim is tied.
        tmp_db.set_embedding(easy_id, [1.0, 0.0, 0.0], "test-model")
        tmp_db.set_embedding(hard_id, [1.0, 0.0, 0.0], "test-model")

        # With target_difficulty=4.5 (matching hard), hard_id should rank first.
        results = tmp_db.retrieve(
            "VR", [1.0, 0.0, 0.0], top_k=2, mmr_lambda=1.0,
            target_difficulty=4.5,
        )
        first_id = results[0][1]["id"]
        assert first_id == hard_id, "target_difficulty=4.5 should prefer the hard doc"

        # With target_difficulty=1.5, easy_id should rank first.
        results = tmp_db.retrieve(
            "VR", [1.0, 0.0, 0.0], top_k=2, mmr_lambda=1.0,
            target_difficulty=1.5,
        )
        first_id = results[0][1]["id"]
        assert first_id == easy_id, "target_difficulty=1.5 should prefer the easy doc"

    def test_user_rating_boosts_retrieval(self, tmp_db: Database):
        """Two docs with identical embeddings rank by user_rating when
        cosine is tied. +1 rating boosts above 0; -1 demotes below."""
        upvoted = _vr_doc(passage_seed="up")
        downvoted = _vr_doc(passage_seed="down")
        neutral = _vr_doc(passage_seed="neutral")
        up_id = tmp_db.add_doc("VR", upvoted)
        down_id = tmp_db.add_doc("VR", downvoted)
        neutral_id = tmp_db.add_doc("VR", neutral)
        for rid in (up_id, down_id, neutral_id):
            tmp_db.set_embedding(rid, [1.0, 0.0, 0.0], "test-model")
        tmp_db.set_user_rating(up_id, 1)
        tmp_db.set_user_rating(down_id, -1)
        # neutral_id has no rating set

        results = tmp_db.retrieve(
            "VR", [1.0, 0.0, 0.0], top_k=3, mmr_lambda=1.0,
        )
        ranked_ids = [d["id"] for _, d in results]
        # Upvoted should rank above neutral, neutral above downvoted.
        assert ranked_ids.index(up_id) < ranked_ids.index(neutral_id)
        assert ranked_ids.index(neutral_id) < ranked_ids.index(down_id)

    def test_set_user_rating_validates_value(self, tmp_db: Database):
        rid = tmp_db.add_doc("VR", _vr_doc())
        with pytest.raises(ValueError):
            tmp_db.set_user_rating(rid, 5)
        with pytest.raises(ValueError):
            tmp_db.set_user_rating(rid, "thumbs_up")  # type: ignore

    def test_promote_to_kb_marks_user_rating(self, tmp_db: Database):
        # Insert a generated row first.
        gen_id = tmp_db.add_generated("VR", _vr_doc(), [])
        ok = tmp_db.promote_to_kb(gen_id)
        assert ok
        # The promoted KB row should have rating=1.
        rows = tmp_db.conn.execute(
            "SELECT user_rating FROM kb WHERE source='generated'"
        ).fetchall()
        assert any(r[0] == 1 for r in rows)

    def test_difficulty_backfill_populates_legacy_rows(self, tmp_path):
        """Pre-migration KB rows should have their kb.difficulty backfilled
        from per-question difficulty fields when the DB is opened."""
        path = str(tmp_path / "legacy.sqlite")
        # Manually insert a doc with no difficulty column populated, then
        # open via Database to trigger migration.
        import sqlite3
        d = _vr_doc()
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE kb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT, source TEXT, data TEXT, embed_text TEXT,
            embedding TEXT, embed_model TEXT, content_hash TEXT,
            created TEXT
        )""")
        conn.execute(
            "INSERT INTO kb (section,data,embed_text) VALUES (?,?,?)",
            ("VR", json.dumps(d), "test"),
        )
        conn.commit()
        conn.close()

        # Open via Database — migration should add column + backfill.
        db = Database(path=path)
        try:
            row = db.conn.execute(
                "SELECT difficulty FROM kb LIMIT 1"
            ).fetchone()
            # All questions in _vr_doc have difficulty=3.0 → avg=3.0
            assert row[0] == pytest.approx(3.0, abs=0.001)
        finally:
            db.close()

    def test_find_near_duplicates_above_threshold(self, tmp_db: Database):
        ids = self._seed_with_embeddings(tmp_db)
        # Query exactly matching doc 0 — should return at least it with sim=1.0
        hits = tmp_db.find_near_duplicates("VR", [1.0, 0.0, 0.0], threshold=0.9)
        assert len(hits) >= 1
        sim, doc = hits[0]
        assert doc["id"] == ids[0]
        assert sim >= 0.9


# ─── Migration: backfill content_hash for legacy rows ─────────────────────────

class TestBackfillMigration:
    def test_legacy_rows_get_content_hash_on_init(self, tmp_path):
        path = str(tmp_path / "legacy.sqlite")
        # Manually create the OLD schema (no content_hash) and insert a row.
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE kb (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            section     TEXT NOT NULL,
            source      TEXT DEFAULT 'manual',
            data        TEXT NOT NULL,
            embed_text  TEXT NOT NULL,
            embedding   TEXT DEFAULT NULL,
            embed_model TEXT DEFAULT NULL,
            created     TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        d = _vr_doc()
        conn.execute(
            "INSERT INTO kb (section,source,data,embed_text) VALUES (?,?,?,?)",
            ("VR", "imported", json.dumps(d), embed_text_for(d)),
        )
        conn.commit()
        conn.close()

        # Now open via Database — migration should run.
        db = Database(path=path)
        try:
            # The legacy row should now be deduplicable.
            assert db.doc_exists("VR", d) is True
            # Re-importing the same doc should skip, not insert a second row.
            rid = db.add_doc("VR", d, source="re-import")
            assert db.count(section="VR") == 1
            assert rid > 0
        finally:
            db.close()
