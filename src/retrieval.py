"""
Advanced RAG retrieval: hybrid search (vector + BM25), Reciprocal Rank Fusion,
section-specific MMR diversity, feedback-weighted boost, and context budget trimming.
"""

import math
import json
import re
from collections import Counter

from src.config import (
    RETRIEVAL_STRATEGIES, CONTEXT_BUDGET, SECTIONS, cosine_sim,
)


# ─── Pure-Python BM25 Fallback ──────────────────────────────────────────────
# Used when FTS5 is unavailable (~30 lines).

def _tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\w+', text.lower())


def _bm25_scores(query_text: str, docs: list, k1: float = 1.5, b: float = 0.75):
    """
    Pure-Python BM25 scoring over documents.
    docs: list of dicts with "id" and "embed_text" keys.
    Returns: list of (doc_id, score) sorted descending.
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens or not docs:
        return []

    # Pre-compute document lengths and term frequencies
    doc_data = []
    for doc in docs:
        tokens = _tokenize(doc.get("embed_text", ""))
        doc_data.append({
            "id": doc["id"],
            "tokens": tokens,
            "tf": Counter(tokens),
            "dl": len(tokens),
        })

    N = len(doc_data)
    avg_dl = sum(d["dl"] for d in doc_data) / N if N else 1

    # Document frequency for query terms
    df = {}
    for t in set(query_tokens):
        df[t] = sum(1 for d in doc_data if t in d["tf"])

    # Score each document
    scored = []
    for d in doc_data:
        score = 0.0
        for t in query_tokens:
            if t not in df or df[t] == 0:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            tf = d["tf"].get(t, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * d["dl"] / avg_dl)
            score += idf * numerator / denominator
        if score > 0:
            scored.append((d["id"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ─── Reciprocal Rank Fusion ─────────────────────────────────────────────────

def reciprocal_rank_fusion(rankings: list, k: int = 60) -> list:
    """
    Merge multiple ranked lists using RRF.
    rankings: list of lists, each [(doc_id, score), ...]
    Returns: [(doc_id, rrf_score)] sorted descending.
    """
    rrf_scores = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    result = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return result


# ─── Maximal Marginal Relevance ─────────────────────────────────────────────

def mmr_select(candidates: list, query_vec: list, top_k: int = 8,
               lambda_param: float = 0.7, diversity_key: str = None):
    """
    Select top_k documents that balance relevance and diversity.
    candidates: list of doc dicts (must have "embedding" key).
    diversity_key: if set, penalizes same-value docs (e.g. same topic_cluster).
    """
    if not candidates:
        return []

    # Filter to candidates with embeddings
    with_emb = [c for c in candidates if c.get("embedding")]
    if not with_emb:
        return candidates[:top_k]

    selected = []
    remaining = list(with_emb)

    for _ in range(min(top_k, len(remaining))):
        best = None
        best_score = -float("inf")

        for doc in remaining:
            # Relevance: similarity to query
            relevance = cosine_sim(query_vec, doc["embedding"])

            # Diversity: max similarity to already-selected docs
            if selected:
                max_sim_to_selected = max(
                    cosine_sim(doc["embedding"], s["embedding"])
                    for s in selected
                )
            else:
                max_sim_to_selected = 0.0

            # Extra diversity penalty for same diversity_key value
            key_penalty = 0.0
            if diversity_key and selected:
                doc_key = doc.get(diversity_key, "")
                if doc_key:
                    same_key_count = sum(
                        1 for s in selected if s.get(diversity_key) == doc_key
                    )
                    key_penalty = 0.1 * same_key_count

            mmr_score = (lambda_param * relevance
                         - (1 - lambda_param) * max_sim_to_selected
                         - key_penalty)

            if mmr_score > best_score:
                best_score = mmr_score
                best = doc

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)

    return selected


# ─── Context Budget Trimming ─────────────────────────────────────────────────

def budget_trim(docs: list, budget_tokens: int) -> list:
    """
    Trim retrieved docs to fit within token budget.
    Drops lowest-scoring docs from the end — never truncates content silently.
    Always keeps at least min_context_docs.
    """
    min_docs = CONTEXT_BUDGET["min_context_docs"]
    chars_per_token = CONTEXT_BUDGET["chars_per_token"]

    total = 0
    kept = []
    for doc in docs:
        doc_chars = len(json.dumps(doc.get("data", {})))
        doc_tokens = doc_chars // chars_per_token
        if total + doc_tokens <= budget_tokens or len(kept) < min_docs:
            kept.append(doc)
            total += doc_tokens
    return kept


# ─── Main Retriever ──────────────────────────────────────────────────────────

class Retriever:
    """
    Advanced RAG retriever with hybrid search, section-specific MMR,
    feedback-weighted boosting, and context budget trimming.
    """

    def __init__(self, db, embedding_engine, feedback_engine=None):
        self.db = db
        self.embeddings = embedding_engine
        self.feedback = feedback_engine

    def retrieve(self, section: str, hint: str = "",
                 budget_tokens: int = None) -> list:
        """
        Full retrieval pipeline:
        1. Vector similarity (top-20)
        2. BM25 keyword search (top-20)
        3. Reciprocal Rank Fusion
        4. Section-specific MMR diversity selection
        5. Feedback-weighted boost
        6. Context budget trimming
        Returns: list of doc dicts, scored and diverse.
        """
        strategy = RETRIEVAL_STRATEGIES.get(section, RETRIEVAL_STRATEGIES["VR"])
        top_k = strategy["top_k"]
        mmr_lambda = strategy["mmr_lambda"]
        diversity_key = strategy.get("diversity_key")

        # Get all docs for section (with embeddings)
        all_docs = self.db.get_all_docs(section)
        docs_by_id = {d["id"]: d for d in all_docs}

        if not all_docs:
            return []

        # 1. Vector similarity search
        try:
            query_vec = self.embeddings.embed_query(section, hint)
        except Exception:
            # Fallback: return random docs
            import random
            random.shuffle(all_docs)
            return all_docs[:top_k]

        vector_results = []
        for doc in all_docs:
            if doc.get("embedding"):
                sim = cosine_sim(query_vec, doc["embedding"])
                vector_results.append((doc["id"], sim))
        vector_results.sort(key=lambda x: x[1], reverse=True)
        vector_top = vector_results[:20]

        # 2. BM25 keyword search
        query_text = f"UCAT {SECTIONS[section]} question"
        if hint.strip():
            query_text += f" {hint.strip()}"

        if self.db.has_fts5:
            bm25_results = self.db.fts5_search(query_text, section, limit=20)
            # FTS5 rank is negative (lower = better), convert to positive scores
            if bm25_results:
                bm25_results = [(doc_id, -rank) for doc_id, rank in bm25_results]
        else:
            # Pure-Python BM25 fallback
            bm25_results = _bm25_scores(query_text, all_docs)[:20]

        # 3. Reciprocal Rank Fusion
        rankings = [vector_top]
        if bm25_results:
            rankings.append(bm25_results)
        fused = reciprocal_rank_fusion(rankings)

        # 4. Apply feedback weights
        if self.feedback:
            weighted = []
            for doc_id, rrf_score in fused:
                doc = docs_by_id.get(doc_id)
                if doc:
                    weight = self.feedback.get_doc_weight(doc)
                    weighted.append((doc_id, rrf_score * weight))
            weighted.sort(key=lambda x: x[1], reverse=True)
            fused = weighted

        # 5. Resolve to full doc objects for MMR
        candidates = []
        for doc_id, score in fused[:20]:  # Top-20 candidates for MMR
            doc = docs_by_id.get(doc_id)
            if doc:
                doc["_rrf_score"] = score
                candidates.append(doc)

        # 6. MMR diversity selection
        selected = mmr_select(
            candidates, query_vec,
            top_k=top_k,
            lambda_param=mmr_lambda,
            diversity_key=diversity_key,
        )

        # 7. For DM with ensure_type_coverage, verify type mix
        if strategy.get("ensure_type_coverage") and section == "DM":
            selected = self._ensure_dm_type_coverage(selected, candidates, query_vec, top_k)

        # 8. Context budget trimming
        if budget_tokens:
            selected = budget_trim(selected, budget_tokens)

        return selected

    def _ensure_dm_type_coverage(self, selected: list, candidates: list,
                                 query_vec: list, top_k: int) -> list:
        """
        For DM section: ensure retrieved examples cover multiple question types
        (syllogism, probability, logical, argument, venn).
        """
        types_covered = set()
        for doc in selected:
            dt = doc.get("data_type", "")
            for t in dt.split(","):
                types_covered.add(t.strip())

        needed_types = {"syllogism", "logical", "probability", "argument", "venn"}
        missing = needed_types - types_covered

        if not missing:
            return selected

        # Try to add docs covering missing types from candidates
        for doc in candidates:
            if doc in selected:
                continue
            dt = doc.get("data_type", "")
            doc_types = {t.strip() for t in dt.split(",")}
            if doc_types & missing:
                if len(selected) < top_k:
                    selected.append(doc)
                else:
                    # Replace the least relevant selected doc
                    selected[-1] = doc
                missing -= doc_types
                if not missing:
                    break

        return selected

    def compute_budget(self, section: str, num_predict: int) -> int:
        """
        Compute available token budget for retrieved context.
        Total context = num_ctx - system_overhead - user_prompt - output_reserve
        """
        overhead = (CONTEXT_BUDGET["system_overhead"]
                    + CONTEXT_BUDGET["user_prompt"]
                    + num_predict)
        # Assume a reasonable total context; actual num_ctx is set dynamically
        # This budget guides how many docs to include
        available = 8000 - overhead  # Conservative estimate
        return max(available, 2000)
