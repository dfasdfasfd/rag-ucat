"""Tests for ucat.retrieval — pure-function vector primitives.

These got flagged in the broader audit as untested. They're trivial to test
(no DB, no LLM) and they're the math floor that db.retrieve and the dedup
threshold check sit on top of, so a regression here would silently
corrupt every generation.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ucat.retrieval import cosine_sim, mmr_select, rerank_by_similarity  # noqa: E402


# ─── cosine_sim ──────────────────────────────────────────────────────────────

class TestCosineSim:
    def test_identical_vectors_are_1(self):
        assert cosine_sim([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_are_0(self):
        assert cosine_sim([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_are_negative_1(self):
        assert cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)

    def test_scale_invariance(self):
        # Cosine similarity ignores magnitude
        a = [1.0, 2.0, 3.0]
        b = [10.0, 20.0, 30.0]  # same direction, 10x magnitude
        assert cosine_sim(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        assert cosine_sim([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == 0.0
        assert cosine_sim([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_known_value(self):
        # cos(45°) ≈ 0.7071
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        assert cosine_sim(a, b) == pytest.approx(1.0 / math.sqrt(2), abs=1e-5)

    def test_higher_dim_known_case(self):
        a = [1.0, 1.0, 1.0, 1.0]
        b = [1.0, 1.0, 0.0, 0.0]
        # |a|=2, |b|=√2, dot=2 → 2 / (2·√2) = 1/√2
        assert cosine_sim(a, b) == pytest.approx(1.0 / math.sqrt(2), abs=1e-5)


# ─── rerank_by_similarity ────────────────────────────────────────────────────

class TestRerankBySimilarity:
    def _docs(self, embeddings):
        return [{"id": i, "embedding": e} for i, e in enumerate(embeddings)]

    def test_sorts_descending_by_similarity(self):
        qvec = [1.0, 0.0]
        docs = self._docs([
            [1.0, 0.0],   # sim = 1.0
            [0.0, 1.0],   # sim = 0.0
            [0.7, 0.3],   # sim = ~0.92
        ])
        ranked = rerank_by_similarity(qvec, docs)
        ids_in_order = [d["id"] for _, d in ranked]
        assert ids_in_order == [0, 2, 1]

    def test_empty_list_returns_empty(self):
        assert rerank_by_similarity([1.0], []) == []

    def test_skips_docs_without_embeddings(self):
        qvec = [1.0, 0.0]
        docs = [
            {"id": 0, "embedding": [1.0, 0.0]},
            {"id": 1, "embedding": None},   # unindexed — should be filtered
            {"id": 2},                       # missing key entirely
            {"id": 3, "embedding": [0.5, 0.5]},
        ]
        ranked = rerank_by_similarity(qvec, docs)
        ids = [d["id"] for _, d in ranked]
        assert 1 not in ids
        assert 2 not in ids
        assert ids == [0, 3]

    def test_preserves_doc_payload_in_tuples(self):
        qvec = [1.0, 0.0]
        docs = [{"id": 99, "embedding": [1.0, 0.0], "extra": "payload"}]
        ranked = rerank_by_similarity(qvec, docs)
        sim, doc = ranked[0]
        assert sim == pytest.approx(1.0)
        assert doc["extra"] == "payload"


# ─── mmr_select ──────────────────────────────────────────────────────────────

class TestMmrSelect:
    def _candidates(self, qvec_sim_doc_pairs):
        return [
            (sim, {"id": i, "embedding": emb})
            for i, (sim, emb) in enumerate(qvec_sim_doc_pairs)
        ]

    def test_empty_pool_returns_empty(self):
        assert mmr_select([1.0], [], top_k=4, lam=0.5) == []

    def test_top_k_zero_returns_empty(self):
        cand = self._candidates([(1.0, [1.0, 0.0])])
        assert mmr_select([1.0, 0.0], cand, top_k=0, lam=0.5) == []

    def test_top_k_greater_than_pool_returns_full_pool(self):
        cand = self._candidates([(1.0, [1.0, 0.0]), (0.5, [0.5, 0.5])])
        result = mmr_select([1.0, 0.0], cand, top_k=10, lam=0.5)
        assert len(result) == 2

    def test_lambda_1_picks_most_similar_first(self):
        # Pure relevance — should equal sorted-by-similarity order. top_k=2
        # forces the algorithm to actually run (top_k>=len short-circuits).
        cand = self._candidates([
            (0.9, [0.9, 0.1]),
            (0.5, [0.5, 0.5]),
            (1.0, [1.0, 0.0]),
        ])
        result = mmr_select([1.0, 0.0], cand, top_k=2, lam=1.0)
        # First pick: highest qsim (1.0) → id=2
        assert result[0][1]["id"] == 2
        # Second pick under lam=1.0: next-highest qsim → id=0 (0.9)
        assert result[1][1]["id"] == 0

    def test_lambda_0_prefers_diversity(self):
        # All three candidates equally relevant to query, but two are identical
        cand = self._candidates([
            (1.0, [1.0, 0.0, 0.0]),
            (1.0, [1.0, 0.0, 0.0]),  # duplicate of first
            (1.0, [0.0, 1.0, 0.0]),  # orthogonal — diverse
        ])
        result = mmr_select([1.0, 0.0, 0.0], cand, top_k=2, lam=0.0)
        # Pure-diversity mode: after picking the first relevant doc, the second
        # pick should maximise distance from it — i.e. the orthogonal one (id=2),
        # not the duplicate (id=1).
        ids = [d["id"] for _, d in result]
        assert ids[0] == 0           # first pick: any relevant doc
        assert ids[1] == 2           # diversifies away from duplicate

    def test_returns_at_most_top_k(self):
        cand = self._candidates([
            (0.9, [0.9, 0.1]), (0.8, [0.8, 0.2]),
            (0.7, [0.7, 0.3]), (0.6, [0.6, 0.4]),
            (0.5, [0.5, 0.5]),
        ])
        result = mmr_select([1.0, 0.0], cand, top_k=3, lam=0.5)
        assert len(result) == 3

    def test_no_duplicate_selection(self):
        # Each doc should appear at most once in the result
        cand = self._candidates([
            (0.9, [0.9, 0.1]),
            (0.8, [0.8, 0.2]),
            (0.7, [0.7, 0.3]),
        ])
        result = mmr_select([1.0, 0.0], cand, top_k=3, lam=0.5)
        ids = [d["id"] for _, d in result]
        assert len(ids) == len(set(ids))
