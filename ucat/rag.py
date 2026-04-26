"""RAG orchestrator — wires retrieval, generation, verification, calibration,
coverage detection, and telemetry into a single end-to-end pipeline.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import (DEFAULT_VERIFY_LLM, DEFAULT_JUDGE2_LLM, IRT_BANDS,
                      Settings, SECTIONS, SECTION_DESC, difficulty_label,
                      SUBTYPES_BY_SECTION)
from .calibration import calibrate_set, difficulty_distance
from .coverage import aggregate_set
from .db import Database, embed_text_for
from .llm import embed_batch, embed_query, generate_structured, merge_usage
from .models import SECTION_MODELS
from .telemetry import emit, logger, trace
from .verification import jury_verify, llm_judge, symbolic_qr_check


def _to_schema_shape(qset: Dict[str, Any]) -> Dict[str, Any]:
    """KB samples and crawler imports store map-typed fields (``options``,
    ``stimulus.rows``) as dicts because that's the canonical downstream shape.
    Anthropic strict mode forbids map types in the schema, so the schema sent
    to Claude uses lists of ``{label, text}`` / ``{name, values}`` objects.
    Convert dict-shape examples to list-shape before rendering them in the
    prompt so the example shape matches the schema shape Claude is asked to
    produce."""
    out = json.loads(json.dumps(qset))  # deep copy; data is JSON-safe
    for q in out.get("questions", []) or []:
        opts = q.get("options")
        if isinstance(opts, dict):
            q["options"] = [{"label": k, "text": v} for k, v in opts.items()]
    stim = out.get("stimulus")
    if isinstance(stim, dict):
        rows = stim.get("rows")
        if isinstance(rows, dict):
            stim["rows"] = [{"name": k, "values": list(vs or [])}
                            for k, vs in rows.items()]
    return out


class RAGEngine:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    @property
    def llm(self):           return self.settings.get("llm")
    @property
    def emb(self):           return self.settings.get("embed")
    @property
    def top_k(self):         return int(self.settings.get("top_k"))
    @property
    def mmr_lambda(self):    return float(self.settings.get("mmr_lambda"))
    @property
    def target_difficulty(self): return float(self.settings.get("target_difficulty"))
    @property
    def verify_enabled(self): return bool(self.settings.get("verify"))
    @property
    def multi_judge(self):   return bool(self.settings.get("multi_judge"))

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_all(self, section: Optional[str] = None,
                   on_progress: Optional[Callable[[int, int], None]] = None) -> int:
        pending = self.db.get_unindexed(section)
        if not pending:
            return 0
        with trace("index", section=section, total=len(pending)) as t:
            from .config import EMBED_BATCH_SIZE
            done = 0
            for i in range(0, len(pending), EMBED_BATCH_SIZE):
                chunk = pending[i:i + EMBED_BATCH_SIZE]
                texts = [d["embed_text"] for d in chunk]
                vecs  = embed_batch(texts, self.emb, "document")
                self.db.set_embeddings_batch([(chunk[j]["id"], vecs[j], self.emb)
                                              for j in range(len(chunk))])
                done += len(chunk)
                if on_progress:
                    on_progress(done, len(pending))
            t["indexed"] = done
            return done

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, section: str, hint: str = ""):
        if hint.strip():
            query = f"UCAT {SECTIONS[section]}: {hint.strip()}"
        else:
            query = f"UCAT {SECTIONS[section]} typical exam-style question"

        with trace("retrieve", section=section, hint_len=len(hint)) as t:
            try:
                qvec = embed_query(query, self.emb)
            except Exception as e:
                logger.warning("Embed query failed (%s) — falling back to random", e)
                import random
                docs = self.db.get_all_docs(section)
                random.shuffle(docs)
                t["fallback"] = "random"
                return None, [(0.0, d) for d in docs[:self.top_k]]
            results = self.db.retrieve(section, qvec, self.top_k, self.mmr_lambda)
            t["retrieved_ids"] = [d["id"] for _, d in results]
            t["retrieved_count"] = len(results)
            return qvec, results

    # ── Generation pipeline ───────────────────────────────────────────────────

    def _system_blocks(self, section: str, retrieved: list,
                        target_difficulty: float,
                        subtype: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build cache-friendly system blocks.

          [0] Frozen role + structural description + retrieved KB     (CACHED)
          [1] Per-request difficulty + variation guidance              (NOT CACHED)

        When ``subtype`` is set, the section-specific block is replaced with
        a lock-in that forces every question (or, for QR, the stimulus chart)
        to that subtype.
        """
        role = (
            f"You are an expert UCAT {SECTIONS[section]} question writer.\n\n"
            f"You generate: {SECTION_DESC[section]}\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "• Return ONLY a JSON object matching the provided schema. No prose, no markdown fences.\n"
            "• Every question's answer must be derivable from the passage / stimulus / rules.\n"
            "• Distractors must be plausible — each wrong option should reflect a realistic mistake.\n"
            "• `options` MUST be a non-empty ARRAY of {\"label\": ..., \"text\": ...} objects, in display order:\n"
            "    - VR / QR multiple choice → 4 items with labels \"A\", \"B\", \"C\", \"D\".\n"
            "    - VR True/False/Can't Tell items → 3 items with labels exactly \"True\", \"False\", \"Can't Tell\".\n"
            "    - DM → 5 items with labels \"A\", \"B\", \"C\", \"D\", \"E\".\n"
            "    - AR test items → 3 items with labels \"Set A\", \"Set B\", \"Neither\".\n"
            "  Never emit an empty `options` array. Never reference option text only inside `explanation`.\n"
            "• `answer` MUST be one of the option labels you emitted in `options[*].label`.\n"
            "• Explanations must show the reasoning a student needs to learn from the question.\n"
            "• Each question must have a `difficulty` field (1.0-5.0 IRT logits) and `coverage` "
            "metadata identifying its topic and scenario type.\n"
        )
        if section == "QR":
            role += (
                "\nCRITICAL — QR stimulus is a STRUCTURED chart spec, not text. "
                "Choose `type` from {bar, line, stacked_bar, pie, table}. "
                "Populate `categories` and `series[]` for bar/line/stacked_bar/pie "
                "(leave `rows` null). "
                "For `table`, populate `rows` as a LIST of {\"name\": <column header>, "
                "\"values\": [<cell values aligned with categories>]} objects "
                "(leave `series` empty). "
                "Include realistic units (e.g. £000s, %, kg). "
                "Make data internally consistent with the questions you write.\n"
            )
            if subtype:
                role += f"\nThe stimulus chart MUST be type: '{subtype}'.\n"
        if section == "AR":
            role += (
                "\nCRITICAL — AR panels are STRUCTURED shape lists. "
                "Each panel contains `shapes[]` with `kind`, `color`, `size`, and "
                "`rotation_deg`. Hidden rules can use any combination of shape kind, "
                "color, size, count, rotation, or position. Provide BOTH a structured "
                "spec for rendering AND a clear English description in the rule field.\n"
            )
        if section == "VR" and subtype:
            kind_reminders = {
                "tfc": (
                    "Set type:'tf' on every question. Use exactly 3 options labelled "
                    '"True", "False", "Can\'t Tell". Each question is a statement the '
                    "student judges as supported / contradicted / not addressed by the "
                    "passage."
                ),
                "main-idea": (
                    "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                    "Each question asks for the main idea, central thesis, best title, "
                    "or overall conclusion of the passage. Distractors should be "
                    "plausible secondary points or over-specific details."
                ),
                "paraphrase": (
                    "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                    "Each question quotes a phrase or sentence from the passage and asks "
                    "which option best restates it. Distractors should be near-paraphrases "
                    "that subtly distort the original meaning."
                ),
                "tone-purpose": (
                    "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                    "Each question asks for the author's tone, attitude, or rhetorical "
                    "purpose (e.g. to argue / inform / caution / evaluate). Options should "
                    "be precise tone words, not synonyms of each other."
                ),
                "inference": (
                    "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                    "Each question asks what can be inferred — a conclusion supported by "
                    "but not stated in the passage. Distractors should be either "
                    "explicitly stated (not inferences) or unsupported."
                ),
            }
            role += (
                f"\nAll 4 questions MUST set `minigame_kind: '{subtype}'`.\n"
                f"{kind_reminders.get(subtype, '')}\n"
            )

        if section == "DM":
            if subtype:
                # Subtype override: replace the variety guidance with a lock-in.
                reminders = {
                    "venn":        "Every question must include a structured `venn` field with 2 or 3 sets.",
                    "probability": "State all probability values clearly; answers must be mathematically verifiable.",
                    "syllogism":   "Premises must be logically sound; conclusions testable.",
                    "logical":     "Each question is a clue-based deduction puzzle. Conclusions must follow from the clues.",
                    "argument":    "Present a clear proposition; options vary in argument strength (strongest/weakest for/against).",
                }
                role += (
                    f"\nAll 5 questions MUST be {subtype} type. "
                    f"Set `type: '{subtype}'` on every question.\n"
                    f"{reminders.get(subtype, '')}\n"
                )
            else:
                role += (
                    "\nFor venn-type DM questions, include a structured `venn` field with "
                    "2 or 3 sets so the diagram can be rendered. For other DM subtypes, "
                    "leave `venn` null.\n"
                    "Aim for variety: include syllogism, logical, venn, probability, AND "
                    "argument subtypes across the 5 questions — one of each is ideal.\n"
                )

        ex_text = ""
        if retrieved:
            blocks = [
                f"--- KNOWLEDGE BASE EXAMPLE {i+1} (similarity {sc:.3f}) ---\n"
                f"{json.dumps(_to_schema_shape(d['data']), indent=2)}"
                for i, (sc, d) in enumerate(retrieved)
            ]
            ex_text = (
                "\n\nThe documents below are gold-standard examples from the user's "
                "knowledge base. They define authoritative format, voice, and topical "
                "range. Your output must mirror their JSON structure exactly.\n\n"
                + "\n\n".join(blocks)
            )

        diff_label = difficulty_label(target_difficulty)
        diff = (
            f"\n\nTARGET DIFFICULTY: {target_difficulty:.1f} logits — {diff_label}\n"
            "Aim for an average set difficulty within ±0.4 of the target. "
            "Include some easier and some harder items around that mean.\n"
        )

        return [
            {"type": "text", "text": role + ex_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": diff},
        ]

    def generate(
        self,
        section: str,
        hint: str = "",
        *,
        subtype: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        on_verify_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        variation_seed: Optional[str] = None,
        force_scenario: Optional[str] = None,
        avoid_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline. Returns a dict with:
          data, retrieved, usage, verdict, coverage, difficulty, dup_warning, row_id

        If `on_verify_complete` is provided AND verify is enabled, the LLM judge
        runs in a background thread: ``generate()`` returns as soon as the set is
        ready, and the callback fires later with an update dict (verdict, usage,
        difficulty, row_id). The fast deterministic symbolic-QR check still runs
        inline so the initial verdict is non-empty for QR sets. With no
        callback, behaviour is unchanged (verify blocks the return).
        """
        target = self.target_difficulty
        async_verify = self.verify_enabled and on_verify_complete is not None

        with trace("rag_generate", section=section, hint=hint[:80],
                   target_difficulty=target,
                   verify=self.verify_enabled,
                   multi_judge=self.multi_judge,
                   async_verify=async_verify,
                   subtype=subtype) as t:

            if on_progress: on_progress("Embedding retrieval query…")
            qvec, retrieved = self.retrieve(section, hint)

            if on_progress: on_progress(f"Retrieved {len(retrieved)} doc(s). Building prompt…")
            system_blocks = self._system_blocks(section, retrieved, target,
                                                  subtype=subtype)

            user_parts = [f"Generate a NEW UCAT {SECTIONS[section]} question set. "]
            if hint.strip():
                user_parts.append(f"Topic focus: {hint.strip()}. ")
            user_parts.append(
                "Content, numbers, scenarios, and passages must be entirely original — "
                "do NOT reuse wording or fact-patterns from the example documents. "
                "Mirror the JSON structure of the examples precisely. "
            )
            if force_scenario:
                user_parts.append(
                    f"DIVERSITY DIRECTIVE: bias every question's "
                    f"`coverage.scenario_type` toward '{force_scenario}' so this set "
                    f"adds variety to recent generations. "
                )
            if avoid_topics:
                shown = ", ".join(f"'{t}'" for t in avoid_topics)
                user_parts.append(
                    f"DIVERSITY DIRECTIVE: avoid these recently-overused topics: "
                    f"{shown}. Pick a different subject. "
                )
            if variation_seed:
                user_parts.append(f"Variation seed for stylistic diversity: {variation_seed}. ")

            if subtype:
                # Look up the human label so the prompt nudge reads naturally.
                label = next(
                    (lbl for v, lbl in SUBTYPES_BY_SECTION.get(section, []) if v == subtype),
                    subtype,
                )
                user_parts.append(f"All questions in this set must be of subtype: {label}. ")

            user_parts.append("Return ONLY the JSON object.")
            user = "".join(user_parts)

            if on_progress: on_progress(f"Generating with {self.llm}…")
            model_cls = SECTION_MODELS[section]
            with trace("generate", section=section, model=self.llm) as gt:
                parsed, gen_usage = generate_structured(
                    system_blocks=system_blocks, user=user,
                    model=self.llm, output_schema=model_cls,
                    on_delta=on_delta, max_tokens=8000,
                )
                gt.update(gen_usage)
            data = parsed.model_dump()
            data["section"] = section

            # Verification — sync path runs LLM judges inline; async path
            # defers them to a background thread (kicked off after the
            # rag_generate trace closes, below).
            verdict_dict: Optional[Dict[str, Any]] = None
            verify_usages: List[Dict[str, Any]] = []
            judge_predictions: Dict[int, float] = {}

            if self.verify_enabled and not async_verify:
                verdict_dict, judge_predictions, verify_usages = self._run_llm_verify(
                    section, data, on_progress=on_progress)

            # Symbolic QR check is fast and deterministic — always runs inline,
            # even on the async path so the initial verdict has *something*.
            if section == "QR":
                sym = symbolic_qr_check(data)
                if verdict_dict is None: verdict_dict = {}
                verdict_dict["symbolic_qr"] = sym
                if sym.get("disagreed"):
                    verdict_dict["overall_correct"] = False

            if async_verify:
                if verdict_dict is None: verdict_dict = {}
                verdict_dict["pending"] = True
                if on_progress: on_progress("Verifying answers in background…")

            # Calibrated difficulty using all signals.
            if on_progress: on_progress("Calibrating difficulty…")
            cal = calibrate_set(data.get("questions") or [], section,
                                  judge_predictions=judge_predictions)
            data["calibrated_difficulty"] = cal
            set_difficulty = cal.get("set_difficulty", target)

            # Bias / coverage tags.
            if on_progress: on_progress("Analysing coverage and bias…")
            coverage_dict = aggregate_set(data)

            # Combined usage.
            total_usage = merge_usage(gen_usage, *verify_usages)
            t.update({
                "input_tokens": total_usage["input_tokens"],
                "output_tokens": total_usage["output_tokens"],
                "cache_read_input_tokens": total_usage["cache_read_input_tokens"],
                "cache_creation_input_tokens": total_usage["cache_creation_input_tokens"],
                "cost_usd": total_usage["cost_usd"],
                "set_difficulty": set_difficulty,
                "difficulty_off_target": round(difficulty_distance(target, set_difficulty), 2),
                "verdict_overall_correct": (verdict_dict or {}).get("overall_correct"),
                "coverage_flags": coverage_dict.get("flags", []),
                "retrieved_ids": [d["id"] for _, d in retrieved],
            })

            # Dedup detection against KB.
            dup_warning = None
            if qvec is not None:
                try:
                    gen_text = embed_text_for(data)
                    gen_vec  = embed_query(gen_text, self.emb)
                    near = self.db.find_near_duplicates(section, gen_vec)
                    if near:
                        top_sim = max(s for s, _ in near)
                        dup_warning = (f"Similar to {len(near)} existing doc(s) "
                                         f"(best sim {top_sim:.3f})")
                        emit("near_duplicate_detected", section=section,
                              count=len(near), top_sim=top_sim)
                except Exception:
                    pass

            ctx_ids = [d["id"] for _, d in retrieved]
            row_id = self.db.add_generated(section, data, ctx_ids,
                                   usage=total_usage,
                                   verdict=verdict_dict,
                                   coverage=coverage_dict,
                                   difficulty=set_difficulty)

        # Spawn the async verify worker AFTER the trace closes so
        # rag_generate's elapsed_ms reflects the user-visible wait. The worker
        # has its own `verify_async` trace span.
        if async_verify:
            threading.Thread(
                target=self._async_verify_worker,
                args=(row_id, section, data, gen_usage, target,
                      verdict_dict, on_verify_complete),
                daemon=True,
            ).start()

        return {
            "data":       data,
            "retrieved":  retrieved,
            "usage":      total_usage,
            "verdict":    verdict_dict,
            "coverage":   coverage_dict,
            "difficulty": cal,
            "dup_warning": dup_warning,
            "row_id":     row_id,
        }

    # ── Verification helpers ─────────────────────────────────────────────────

    def _run_llm_verify(
        self,
        section: str,
        data: Dict[str, Any],
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Dict[str, Any], Dict[int, float], List[Dict[str, Any]]]:
        """Run the LLM-judge step (single or jury). Returns
        ``(verdict_dict, judge_predictions, verify_usages)``. Does NOT touch
        symbolic_qr — caller stitches that in. Single-judge errors are caught
        and surfaced as a soft "low confidence" verdict so generation never
        fails because verify did.
        """
        if self.multi_judge:
            if on_progress: on_progress("Multi-judge jury verifying…")
            judges = [DEFAULT_VERIFY_LLM, DEFAULT_JUDGE2_LLM, self.llm]
            jury = jury_verify(section, data, judges=judges)
            verdict_dict: Dict[str, Any] = {
                "mode": "jury",
                "judges": jury["judges"],
                "individual": jury["individual"],
                "failed_judges": jury.get("failed_judges") or [],
                "overall_correct": jury["overall_correct"],
                "unanimous": jury["unanimous"],
                "flagged_questions": jury["flagged_questions"],
            }
            return verdict_dict, jury["judge_predictions"], jury["usage"]

        if on_progress: on_progress("Verifying answers (Haiku)…")
        try:
            v, vu = llm_judge(section, data, DEFAULT_VERIFY_LLM)
            verdict_dict = {"mode": "single",
                              "judge": DEFAULT_VERIFY_LLM,
                              **v.model_dump()}
            judge_predictions = {pq.number: pq.difficulty for pq in v.per_question}
            return verdict_dict, judge_predictions, [vu]
        except Exception as e:
            return ({"mode": "single", "error": str(e),
                      "overall_correct": True, "confidence": "low"},
                    {}, [])

    def _async_verify_worker(
        self,
        row_id: int,
        section: str,
        data: Dict[str, Any],
        gen_usage: Dict[str, Any],
        target: float,
        base_verdict: Optional[Dict[str, Any]],
        on_complete: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Background worker: run LLM verify, recalibrate with judge difficulty
        signals, persist the updated row, then fire ``on_complete`` with a dict
        the UI can splice into its existing state.
        """
        try:
            with trace("verify_async", section=section, row_id=row_id,
                       model=self.llm, multi_judge=self.multi_judge) as t:
                verdict_dict, judge_predictions, verify_usages = \
                    self._run_llm_verify(section, data)

                # Re-stitch the symbolic_qr verdict that ran inline. If symbolic
                # disagreed, that wins — we trust deterministic over LLM.
                if base_verdict and "symbolic_qr" in base_verdict:
                    verdict_dict["symbolic_qr"] = base_verdict["symbolic_qr"]
                    if base_verdict["symbolic_qr"].get("disagreed"):
                        verdict_dict["overall_correct"] = False

                cal = calibrate_set(data.get("questions") or [], section,
                                      judge_predictions=judge_predictions)
                set_difficulty = cal.get("set_difficulty", target)
                verify_usage = merge_usage(*verify_usages) if verify_usages else None
                total_usage = merge_usage(gen_usage, *verify_usages)

                self.db.update_generated_verdict(
                    row_id,
                    verdict=verdict_dict,
                    usage=total_usage,
                    difficulty=set_difficulty,
                )

                # The verify_async span reports VERIFY-only tokens/cost so
                # downstream analysis can separate gen-time from verify-time
                # spend. The merged total_usage is on the row in the db.
                vu = verify_usage or {}
                t.update({
                    "input_tokens": vu.get("input_tokens", 0),
                    "output_tokens": vu.get("output_tokens", 0),
                    "cache_read_input_tokens": vu.get("cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": vu.get("cache_creation_input_tokens", 0),
                    "cost_usd": vu.get("cost_usd", 0.0),
                    "set_difficulty": set_difficulty,
                    "difficulty_off_target": round(difficulty_distance(target, set_difficulty), 2),
                    "verdict_overall_correct": verdict_dict.get("overall_correct"),
                })

            try:
                on_complete({
                    "row_id":       row_id,
                    "verdict":      verdict_dict,
                    "usage":        total_usage,
                    "verify_usage": verify_usage,  # delta only — already-counted gen_usage excluded
                    "difficulty":   cal,
                })
            except Exception:
                logger.exception("on_verify_complete callback failed")
        except Exception:
            logger.exception("Async verify worker failed for row_id=%s", row_id)
