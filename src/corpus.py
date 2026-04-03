"""
Corpus gap analysis: pure-Python k-means clustering on embeddings,
topic coverage mapping, and underrepresentation detection.
"""

import random
import re
from collections import Counter

from src.config import SECTIONS, cosine_sim


def cluster_docs(embeddings: list, k: int = 8, max_iter: int = 20) -> tuple:
    """
    Simple k-means clustering on embedding vectors.
    Returns: (assignments, centroids)
      assignments: list of cluster indices (one per embedding)
      centroids: list of centroid vectors
    """
    if not embeddings or k <= 0:
        return [], []

    k = min(k, len(embeddings))
    if k == 0:
        return [], []

    # Random initialization from existing points
    centroids = [list(e) for e in random.sample(embeddings, k)]
    dim = len(centroids[0])
    assignments = [0] * len(embeddings)

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        changed = False
        for i, emb in enumerate(embeddings):
            best_c = 0
            best_sim = -1
            for c in range(len(centroids)):
                sim = cosine_sim(emb, centroids[c])
                if sim > best_sim:
                    best_sim = sim
                    best_c = c
            if assignments[i] != best_c:
                changed = True
                assignments[i] = best_c

        if not changed:
            break

        # Recompute centroids as mean of assigned vectors
        for c in range(len(centroids)):
            members = [embeddings[i] for i, a in enumerate(assignments) if a == c]
            if members:
                centroids[c] = [
                    sum(m[d] for m in members) / len(members)
                    for d in range(dim)
                ]

    return assignments, centroids


def extract_topic_keywords(texts: list, top_n: int = 5) -> list:
    """Extract the most common meaningful words from a list of texts."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "of", "in", "to",
        "for", "with", "on", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "and",
        "but", "or", "not", "no", "this", "that", "these", "those", "it",
        "its", "he", "she", "they", "we", "you", "which", "who", "whom",
        "what", "where", "when", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "than", "too",
        "very", "just", "also", "ucat", "question", "answer", "option",
        "represent", "retrieval", "reasoning", "verbal", "following",
    }

    words = Counter()
    for text in texts:
        tokens = re.findall(r'[a-z]{3,}', text.lower())
        for t in tokens:
            if t not in stop_words:
                words[t] += 1

    return [w for w, _ in words.most_common(top_n)]


class CorpusAnalyzer:
    """Analyzes KB corpus for coverage gaps and topic distribution."""

    def __init__(self, db):
        self.db = db

    def analyze_section(self, section: str, k: int = 6) -> dict:
        """
        Cluster documents in a section and report coverage.
        Returns dict with clusters, gaps, and recommendations.
        """
        docs = self.db.get_all_docs(section)
        if not docs:
            return {
                "section": section,
                "total_docs": 0,
                "clusters": [],
                "gaps": [f"No documents in {SECTIONS[section]}. Import some questions first."],
                "recommendations": [],
            }

        # Get docs with embeddings
        embedded_docs = [d for d in docs if d.get("embedding")]
        if len(embedded_docs) < 2:
            return {
                "section": section,
                "total_docs": len(docs),
                "indexed_docs": len(embedded_docs),
                "clusters": [],
                "gaps": ["Not enough indexed documents for clustering. Index your KB first."],
                "recommendations": [],
            }

        # Cluster
        embeddings = [d["embedding"] for d in embedded_docs]
        actual_k = min(k, len(embeddings))
        assignments, centroids = cluster_docs(embeddings, k=actual_k)

        # Build cluster summaries
        clusters = []
        for c in range(actual_k):
            cluster_docs_list = [
                embedded_docs[i] for i, a in enumerate(assignments) if a == c
            ]
            if not cluster_docs_list:
                continue

            # Extract topic keywords from embed_text
            texts = [d.get("embed_text", "") for d in cluster_docs_list]
            keywords = extract_topic_keywords(texts)

            # Count data types
            type_counts = Counter(d.get("data_type", "unknown") for d in cluster_docs_list)

            clusters.append({
                "id": c,
                "size": len(cluster_docs_list),
                "keywords": keywords,
                "label": ", ".join(keywords[:3]) if keywords else f"Cluster {c+1}",
                "data_types": dict(type_counts),
                "doc_ids": [d["id"] for d in cluster_docs_list],
            })

        clusters.sort(key=lambda c: c["size"], reverse=True)

        # Detect gaps
        gaps = []
        recommendations = []

        # Check for topic imbalance
        if clusters:
            sizes = [c["size"] for c in clusters]
            max_size = max(sizes)
            total = sum(sizes)

            for cl in clusters:
                pct = (cl["size"] * 100) // total
                if pct > 50:
                    gaps.append(
                        f"{pct}% of docs are about '{cl['label']}' — "
                        f"consider adding different topics"
                    )

            # Recommend underrepresented clusters
            for cl in clusters:
                if cl["size"] <= 1:
                    recommendations.append(
                        f"Only {cl['size']} doc(s) about '{cl['label']}' — "
                        f"add more to improve diversity"
                    )

        # Section-specific gap checks
        if section == "DM":
            all_types = set()
            for doc in docs:
                dt = doc.get("data_type", "")
                for t in dt.split(","):
                    all_types.add(t.strip())
            expected = {"syllogism", "logical", "probability", "argument", "venn"}
            missing = expected - all_types
            if missing:
                gaps.append(f"Missing DM question types: {', '.join(missing)}")

        elif section == "QR":
            all_types = {d.get("data_type", "") for d in docs}
            if len(all_types) < 3:
                recommendations.append(
                    "Add QR questions with different calculation types "
                    "(percentage, ratio, rate, area)"
                )

        return {
            "section": section,
            "section_name": SECTIONS[section],
            "total_docs": len(docs),
            "indexed_docs": len(embedded_docs),
            "clusters": clusters,
            "gaps": gaps,
            "recommendations": recommendations,
        }

    def analyze_all(self) -> dict:
        """Analyze all sections and return combined report."""
        results = {}
        for section in SECTIONS:
            results[section] = self.analyze_section(section)
        return results

    def coverage_summary(self) -> str:
        """Generate a human-readable coverage summary."""
        report = self.analyze_all()
        lines = ["═" * 50, "  Knowledge Base Coverage Analysis", "═" * 50, ""]

        for section, analysis in report.items():
            name = SECTIONS[section]
            total = analysis["total_docs"]
            indexed = analysis.get("indexed_docs", 0)
            clusters = analysis.get("clusters", [])

            lines.append(f"▸ {name} ({section}): {total} docs ({indexed} indexed)")

            if clusters:
                for cl in clusters[:5]:
                    lines.append(f"    • {cl['label']}: {cl['size']} docs")

            for gap in analysis.get("gaps", []):
                lines.append(f"    ⚠ {gap}")

            for rec in analysis.get("recommendations", []):
                lines.append(f"    → {rec}")

            lines.append("")

        return "\n".join(lines)
