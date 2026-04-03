"""
Generation pipeline: retrieve → budget → stream-generate → validate → retry.
Orchestrates the full RAG question generation with quality gates.
"""

import json

from src.config import SECTION_GEN_PARAMS, SECTIONS, CONTEXT_BUDGET
from src.prompts import PromptBuilder
from src.retrieval import budget_trim


class GenerationPipeline:
    """
    Full question generation pipeline with streaming, retry,
    context budgeting, and quality gates.
    """

    def __init__(self, db, ollama_client, embedding_engine,
                 retriever, quality_pipeline, feedback_engine=None):
        self.db = db
        self.ollama = ollama_client
        self.embeddings = embedding_engine
        self.retriever = retriever
        self.quality = quality_pipeline
        self.feedback = feedback_engine
        self.prompts = PromptBuilder()

    def generate(self, section: str, hint: str = "",
                 difficulty: int = None, on_progress=None,
                 stream: bool = False, on_token=None,
                 abort_flag=None, max_retries: int = 3) -> tuple:
        """
        Full generation pipeline with retry and quality gates.

        Returns: (data, retrieved_docs, quality_report)
        Raises RuntimeError after max_retries failures.
        """
        params = SECTION_GEN_PARAMS[section]
        model = params["preferred_llm"]
        gen_options = {
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "num_predict": params["num_predict"],
        }

        # 1. Retrieve with section-specific strategy
        if on_progress:
            on_progress("Retrieving relevant documents...")
        budget = self._compute_context_budget(params["num_predict"])
        retrieved = self.retriever.retrieve(section, hint, budget_tokens=budget)

        if on_progress:
            on_progress(f"Retrieved {len(retrieved)} reference document(s). Building prompt...")

        # 2. Build structured prompt
        system, user, version_hash = self.prompts.build(
            section, retrieved, hint, difficulty
        )

        # 3. Generate with retry loop
        last_errors = []
        for attempt in range(max_retries):
            if on_progress:
                if attempt == 0:
                    on_progress(f"Generating with {model}...")
                else:
                    on_progress(f"Retry {attempt}/{max_retries-1}: regenerating...")

            # Generate (streaming or non-streaming)
            try:
                if stream and on_token:
                    if abort_flag:
                        raw = self.ollama.generate_stream_abortable(
                            system, user, model, gen_options,
                            on_token=on_token, abort_flag=abort_flag
                        )
                    else:
                        raw = self.ollama.generate_stream(
                            system, user, model, gen_options,
                            on_token=on_token
                        )
                else:
                    raw = self.ollama.generate(
                        system, user, model, gen_options
                    )
            except Exception as e:
                last_errors.append(f"Generation failed: {str(e)}")
                continue

            # Check if aborted
            if abort_flag and abort_flag():
                raise RuntimeError("Generation aborted by user")

            # 4. Parse JSON (format:json should guarantee valid JSON)
            if on_progress:
                on_progress("Parsing output...")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                last_errors.append("Output is not valid JSON despite format:json")
                if attempt < max_retries - 1:
                    system, user = self.prompts.build_retry(
                        system, user, raw[:2000],
                        ["Output was not valid JSON. Return ONLY a JSON object."]
                    )
                continue

            data["section"] = section

            # 5. Schema + content validation
            if on_progress:
                on_progress("Validating output...")
            errors = self.quality.validate(data, section)
            schema_errors = [e for e in errors if not e.startswith("Contains")]

            if schema_errors and attempt < max_retries - 1:
                last_errors.extend(schema_errors)
                system, user = self.prompts.build_retry(
                    system, user, raw[:2000], schema_errors
                )
                continue

            # 6. Dedup checks
            if on_progress:
                on_progress("Checking for duplicates...")
            try:
                embedding = self.embeddings.embed_document(data, section)
            except Exception:
                embedding = None

            if embedding:
                dedup = self.quality.check_dedup(embedding, section)

                if dedup.get("kb_duplicate") and attempt < max_retries - 1:
                    last_errors.append(
                        f"Too similar to KB doc (similarity: {dedup['kb_similarity']:.3f})"
                    )
                    system, user = self.prompts.build_dedup_retry(system, user, "KB")
                    continue

                if dedup.get("session_duplicate") and attempt < max_retries - 1:
                    last_errors.append(
                        f"Too similar to recent generation (similarity: {dedup['session_similarity']:.3f})"
                    )
                    system, user = self.prompts.build_dedup_retry(system, user, "session")
                    continue

            # 7. Quality scoring
            if on_progress:
                on_progress("Scoring quality...")
            report = self.quality.full_assessment(data, section, embedding)

            # 8. Store results
            ctx_ids = [d["id"] for d in retrieved]
            gen_id = self.db.add_generated(
                section, data, ctx_ids, prompt_version=version_hash
            )

            # Store quality log
            self.db.add_quality_log(
                gen_id=gen_id,
                rule_score=report["rule_score"],
                llm_score=report["llm_score"],
                final_score=report["final_score"],
                dedup_kb_sim=report.get("dedup", {}).get("kb_similarity", 0.0),
                dedup_session_sim=report.get("dedup", {}).get("session_similarity", 0.0),
                format_valid=report["format_valid"],
                errors=report.get("errors", []),
            )

            # 9. Add to session dedup cache
            if embedding:
                self.quality.session_cache.add(section, embedding)

            # 10. Record feedback
            if self.feedback:
                self.feedback.record_generation(ctx_ids, data, report)

            if on_progress:
                on_progress("Done!")

            report["gen_id"] = gen_id
            report["prompt_version"] = version_hash
            return data, retrieved, report

        # All retries exhausted
        raise RuntimeError(
            f"Failed after {max_retries} attempts.\n"
            f"Errors: {'; '.join(last_errors[-3:])}\n\n"
            "Suggestions:\n"
            "• Add more documents to the knowledge base\n"
            "• Try a different topic hint\n"
            "• Try generating again"
        )

    def _compute_context_budget(self, num_predict: int) -> int:
        """Compute token budget available for retrieved context."""
        overhead = (CONTEXT_BUDGET["system_overhead"]
                    + CONTEXT_BUDGET["user_prompt"]
                    + num_predict)
        # Conservative: assume we can push context to ~12k tokens total
        total_available = 12000
        return max(total_available - overhead, 2000)
