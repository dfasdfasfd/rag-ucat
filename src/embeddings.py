"""
Embedding engine with instruction-prefixed embeddings for mxbai-embed-large.
Full-text extraction (no truncation), batch operations, and model mismatch detection.
"""

from src.config import DEFAULT_EMBED, EMBED_INSTRUCTIONS, SECTIONS


class EmbeddingEngine:
    """Manages document embedding with instruction prefixes and model tracking."""

    def __init__(self, ollama_client, db, model: str = DEFAULT_EMBED):
        self.ollama = ollama_client
        self.db = db
        self.model = model

    # ─── Embed Text Construction ─────────────────────────────────────────────

    @staticmethod
    def embed_text_for(data: dict, section: str) -> str:
        """
        Build rich embedding text with section-specific instruction prefix.
        NO truncation — uses full passage, ALL questions, and ALL explanations.
        mxbai-embed-large performs measurably better with instruction prefixes.
        """
        prefix = EMBED_INSTRUCTIONS.get(section, "Represent this text for retrieval: ")
        parts = []

        # Full passage / stimulus / set descriptions
        for key in ("passage", "stimulus", "scenario"):
            if key in data:
                parts.append(str(data[key]))

        # ALL questions with their text and explanations
        for q in data.get("questions", []):
            text = q.get("text", "")
            if text:
                parts.append(text)
            explanation = q.get("explanation", "")
            if explanation:
                parts.append(explanation)
            # Include question type for DM diversity
            qtype = q.get("type", "")
            if qtype:
                parts.append(f"type: {qtype}")

        content = "\n".join(parts)
        return prefix + content

    # ─── Single Document Embedding ───────────────────────────────────────────

    def embed_text(self, text: str) -> list:
        """Embed raw text string using the configured model."""
        return self.ollama.embed(text, self.model)

    def embed_document(self, data: dict, section: str) -> list:
        """Build embed text from document data and embed it."""
        text = self.embed_text_for(data, section)
        return self.embed_text(text)

    # ─── Index Operations ────────────────────────────────────────────────────

    def index_doc(self, doc_id: int, embed_text: str):
        """Embed and store a single document."""
        vec = self.ollama.embed(embed_text, self.model)
        self.db.set_embedding(doc_id, vec, self.model)

    def index_all(self, section=None, on_progress=None) -> int:
        """Index all unindexed documents. Returns count of newly indexed."""
        pending = self.db.get_unindexed(section)
        for i, doc in enumerate(pending):
            if on_progress:
                on_progress(i, len(pending), "indexing")
            self.index_doc(doc["id"], doc["embed_text"])
        return len(pending)

    def reindex_mismatched(self, section=None, on_progress=None) -> int:
        """Re-embed documents that were embedded with a different model."""
        mismatched = self.db.get_model_mismatched(self.model, section)
        for i, doc in enumerate(mismatched):
            if on_progress:
                on_progress(i, len(mismatched), "re-indexing")
            self.index_doc(doc["id"], doc["embed_text"])
        return len(mismatched)

    def index_and_reindex(self, section=None, on_progress=None) -> dict:
        """Index unindexed + re-embed mismatched. Returns counts."""
        new_count = self.index_all(section, on_progress)
        reindex_count = self.reindex_mismatched(section, on_progress)
        return {"new": new_count, "reindexed": reindex_count}

    # ─── Query Embedding ─────────────────────────────────────────────────────

    def embed_query(self, section: str, hint: str = "") -> list:
        """Embed a retrieval query with section-appropriate prefix."""
        prefix = EMBED_INSTRUCTIONS.get(section, "")
        query = f"UCAT {SECTIONS[section]} question"
        if hint.strip():
            query += f" about {hint.strip()}"
        return self.ollama.embed(prefix + query, self.model)

    # ─── Data Type Inference ─────────────────────────────────────────────────

    @staticmethod
    def infer_data_type(data: dict, section: str) -> str:
        """Infer the data_type tag for a document based on its content."""
        if section == "DM":
            types_found = set()
            for q in data.get("questions", []):
                qtype = q.get("type", "").lower()
                if qtype:
                    types_found.add(qtype)
            return ",".join(sorted(types_found)) if types_found else "mixed"

        elif section == "QR":
            stimulus = str(data.get("stimulus", "")).lower()
            if any(w in stimulus for w in ["percent", "%", "percentage"]):
                return "percentage"
            elif any(w in stimulus for w in ["ratio", "proportion"]):
                return "ratio"
            elif any(w in stimulus for w in ["rate", "speed", "per hour", "per minute"]):
                return "rate"
            elif any(w in stimulus for w in ["area", "volume", "length", "width"]):
                return "area"
            return "general"

        elif section == "SJT":
            scenario = str(data.get("scenario", "")).lower()
            if any(w in scenario for w in ["team", "colleague", "coworker", "staff"]):
                return "teamwork"
            elif any(w in scenario for w in ["patient", "care", "treatment", "clinical"]):
                return "patient-care"
            elif any(w in scenario for w in ["ethic", "honest", "integrity", "confidential"]):
                return "ethics"
            elif any(w in scenario for w in ["conflict", "disagree", "complaint", "angry"]):
                return "conflict"
            return "general"

        elif section == "VR":
            passage = str(data.get("passage", "")).lower()[:200]
            # Simple topic detection from first 200 chars
            if any(w in passage for w in ["cell", "gene", "species", "organism", "biolog"]):
                return "science-biology"
            elif any(w in passage for w in ["atom", "molecule", "chemical", "physic"]):
                return "science-physics-chemistry"
            elif any(w in passage for w in ["history", "century", "ancient", "war", "empire"]):
                return "humanities-history"
            elif any(w in passage for w in ["society", "culture", "politic", "econom"]):
                return "social-sciences"
            elif any(w in passage for w in ["art", "music", "literature", "novel", "poem"]):
                return "humanities-arts"
            elif any(w in passage for w in ["technology", "computer", "digital", "internet"]):
                return "technology"
            elif any(w in passage for w in ["health", "disease", "medical", "patient"]):
                return "health-medicine"
            return "general"

        return "unknown"
