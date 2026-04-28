"""Vector retrieval primitives: cosine similarity + Maximal Marginal Relevance.

These are pure functions on lists/arrays — no DB or LLM dependencies — so they
can be unit-tested in isolation.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. NumPy-accelerated; falls back to pure Python."""
    if _HAS_NUMPY:
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        na = float(np.linalg.norm(a_arr))
        nb = float(np.linalg.norm(b_arr))
        return float(np.dot(a_arr, b_arr) / (na * nb)) if na and nb else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    ma  = math.sqrt(sum(x * x for x in a))
    mb  = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


def mmr_select(
    qvec: Sequence[float],
    candidates: List[Tuple[float, dict]],
    top_k: int,
    lam: float,
) -> List[Tuple[float, dict]]:
    """
    Maximal Marginal Relevance — picks `top_k` items balancing relevance to
    query (`lam`) vs diversity from already-picked items (`1-lam`).

    Each candidate is `(query_similarity, doc_dict)`. Each `doc_dict` must have
    an `embedding` field for diversity comparison.
    """
    if not candidates or top_k <= 0:
        return []
    if top_k >= len(candidates):
        return list(candidates)

    selected: List[Tuple[float, dict]] = []
    pool = list(candidates)
    while pool and len(selected) < top_k:
        best_idx = 0
        best_score = -1e9
        for i, (qsim, doc) in enumerate(pool):
            if selected:
                div = max(cosine_sim(doc["embedding"], s[1]["embedding"]) for s in selected)
            else:
                div = 0.0
            score = lam * qsim - (1 - lam) * div
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(pool.pop(best_idx))
    return selected


def rerank_by_similarity(qvec: Sequence[float], docs: Iterable[dict]) -> List[Tuple[float, dict]]:
    """Sort docs by cosine similarity to qvec (descending). Skips docs without embeddings."""
    return sorted(
        ((cosine_sim(qvec, d["embedding"]), d) for d in docs if d.get("embedding")),
        key=lambda x: x[0],
        reverse=True,
    )
