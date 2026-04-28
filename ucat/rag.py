"""RAG orchestrator — wires retrieval, generation, verification, calibration,
coverage detection, and telemetry into a single end-to-end pipeline.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .config import (DEFAULT_VERIFY_LLM, DEFAULT_JUDGE2_LLM, IRT_BANDS,
                       PRIMARY_JUDGE_BY_SECTION,
                      Settings, SECTIONS, SECTION_DESC, difficulty_label,
                      SUBTYPES_BY_SECTION)
from .calibration import calibrate_set, difficulty_distance
from .coverage import aggregate_set
from .db import Database, embed_text_for
from .llm import embed_batch, embed_doc, embed_query, generate_structured, merge_usage
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


def _detect_subtype_drift(section: str, data: Dict[str, Any],
                            subtype: Optional[str]) -> Optional[str]:
    """Return a human-readable drift message if the parsed set doesn't match
    the requested subtype, else None.

    - DM → checks Question.type on every question
    - VR → checks Question.minigame_kind on every question
    - QR → checks QRChart.type on the stimulus
    - AR → no subtype targeting; always returns None
    - subtype is None → always returns None
    """
    if not subtype:
        return None

    if section == "QR":
        actual = (data.get("stimulus") or {}).get("type")
        if actual != subtype:
            return f"Asked {subtype}, got chart type {actual!r}"
        return None

    if section == "AR":
        return None

    field = "minigame_kind" if section == "VR" else "type"
    actuals = [q.get(field) for q in data.get("questions", [])]
    if not all(a == subtype for a in actuals):
        return f"Asked {subtype}, got {actuals}"
    return None


class RAGEngine:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        # Single-worker executor for async verify tasks. Replaces the previous
        # per-call `threading.Thread` spawn + lets us cancel pending verifies
        # and shut down cleanly. `max_workers=1` is intentional: verify tasks
        # don't benefit from running in parallel (each one already
        # parallelises 2 judges internally via verification.py's TPE), and a
        # single worker avoids surprising concurrent DB writes on the
        # generated table.
        self._verify_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rag-verify"
        )
        # Tracks not-yet-completed verify futures so callers can cancel
        # pending ones (only effective for futures that haven't started yet —
        # in-flight Claude calls run to completion).
        self._verify_futures: Set[Future] = set()
        self._verify_lock = threading.Lock()
        self._cancelled = False

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

    def retrieve(
        self,
        section: str,
        hint: str = "",
        subtype: Optional[str] = None,
        target_difficulty: Optional[float] = None,
    ):
        # Bake subtype into the query string so the embedding anchor lands
        # closer to subtype-matching KB docs even before the SQL pre-filter
        # narrows the pool. (Example: "UCAT Decision Making venn: ..." pulls
        # cleaner venn-heavy retrievals than "UCAT Decision Making: ...".)
        prefix = f"UCAT {SECTIONS[section]}"
        if subtype:
            prefix += f" {subtype}"
        if hint.strip():
            query = f"{prefix}: {hint.strip()}"
        else:
            query = f"{prefix} typical exam-style question"

        with trace("retrieve", section=section, hint_len=len(hint), subtype=subtype) as t:
            try:
                qvec = embed_query(query, self.emb)
            except Exception as e:
                logger.warning("Embed query failed (%s) — falling back to random", e)
                import random
                docs = self.db.get_all_docs(section)
                random.shuffle(docs)
                t["fallback"] = "random"
                return None, [(0.0, d) for d in docs[:self.top_k]]
            results = self.db.retrieve(
                section, qvec, self.top_k, self.mmr_lambda,
                subtype=subtype, target_difficulty=target_difficulty,
            )
            t["retrieved_ids"] = [d["id"] for _, d in results]
            t["retrieved_count"] = len(results)
            return qvec, results

    # ── Generation pipeline ───────────────────────────────────────────────────

    def _system_blocks(self, section: str, retrieved: list,
                        target_difficulty: float,
                        subtype: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build cache-friendly system blocks.

          [0] Frozen role + structural description           (CACHED — stable)
          [1] Retrieved KB examples                          (NOT CACHED — varies per call)
          [2] Per-request difficulty + variation guidance    (NOT CACHED)

        Why the split: putting `cache_control` on a block that includes the
        retrieved examples invalidates the cache on almost every call (the
        retrieved set changes with the query). Caching only the stable role
        block lets the ~2KB role text hit cache on every subsequent call
        within a section, saving ~90% of the input tokens for that block.

        When ``subtype`` is set, the role block is augmented with a
        lock-in that forces every question (or, for QR, the stimulus chart)
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
            "    - VR multiple choice → 4 items with labels \"A\", \"B\", \"C\", \"D\".\n"
            "    - VR True/False/Can't Tell items → 3 items with labels exactly \"True\", \"False\", \"Can't Tell\".\n"
            "    - QR → 5 numerical items with labels \"A\", \"B\", \"C\", \"D\", \"E\".\n"
            "    - DM → 5 items with labels \"A\", \"B\", \"C\", \"D\", \"E\".\n"
            "    - SJT → 4 Likert-scale items with labels \"A\", \"B\", \"C\", \"D\".\n"
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

        if section == "SJT":
            role += (
                "\nCRITICAL — SJT scoring is rubric-based, not single-correct-answer. "
                "Set `answer` to the option label that best aligns with UCAT's marking "
                "guidance — weighted toward professionalism, patient safety, honesty, "
                "team-working, and scope-of-practice. The answer is the most "
                "appropriate option per UCAT principles, NOT an objectively correct one.\n\n"
                "Likert label rules:\n"
                "  - Appropriateness questions use exactly: 'Very appropriate', "
                "'Appropriate but not ideal', 'Inappropriate but not awful', "
                "'Very inappropriate'.\n"
                "  - Importance questions use exactly: 'Very important', 'Important', "
                "'Of minor importance', 'Not important at all'.\n"
                "Each question's `type` MUST be 'appropriateness' or 'importance'. The "
                "4 questions in a set may mix these types — that's expected.\n\n"
                "Set the SJTSet's `situation_type` field to one of the four UCAT "
                "situation families:\n"
                "  - 'medical_ethics': informed consent, end-of-life care, "
                "confidentiality breaches, patient autonomy.\n"
                "  - 'team_conflict': disagreement with a colleague or senior, peer "
                "behaviour concerns, hierarchy navigation.\n"
                "  - 'boundary_management': scope of practice, dual relationships "
                "with patients, gifts, social-media boundaries.\n"
                "  - 'professional_communication': handover, error reporting, "
                "breaking bad news, disclosure.\n"
                "Avoid scenarios where the 'right' answer is culturally obvious "
                "without UCAT principles.\n"
            )
            if subtype:
                # Subtype lock-in for bulk runs requesting only one type.
                role += (
                    f"\nAll 4 questions MUST set `type: '{subtype}'`.\n"
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
                    "MANDATORY VARIETY — UCAT mixed DM sets feature one of each of the 5 "
                    "subtypes. Set `type` per question as follows:\n"
                    "  • Q1: syllogism\n"
                    "  • Q2: logical (clue-based deduction)\n"
                    "  • Q3: venn (with structured `venn` field)\n"
                    "  • Q4: probability\n"
                    "  • Q5: argument (strongest argument for/against)\n"
                    "Default to this 1-of-each lineup. Only deviate if the user's hint "
                    "explicitly requests otherwise. Do NOT cluster on syllogism — "
                    "Claude's training data is syllogism-heavy and post-hoc coverage "
                    "checks flag any subtype that appears 4+ times in a set.\n"
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

        # Cache only the stable role block. Retrieved examples and the
        # per-call difficulty header sit AFTER the breakpoint so they don't
        # bust the cache when the query changes.
        blocks: List[Dict[str, Any]] = [
            {"type": "text", "text": role, "cache_control": {"type": "ephemeral"}},
        ]
        if ex_text:
            blocks.append({"type": "text", "text": ex_text})
        blocks.append({"type": "text", "text": diff})
        return blocks

    def generate(
        self,
        section: str,
        hint: str = "",
        *,
        subtype: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        on_verify_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
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
            qvec, retrieved = self.retrieve(
                section, hint, subtype=subtype, target_difficulty=target,
            )

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
            # Note: a `variation_seed` kwarg used to live here but it was
            # theatre — Claude has no mechanism to use an opaque hex string
            # as actual sampling entropy. The real diversity levers are
            # `force_scenario`, `avoid_topics`, and the rotating retrieval
            # pool (different examples on each call).

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
            # Adaptive thinking on for sections that require multi-step
            # reasoning. DM venn/probability questions and QR multi-step
            # calculations show meaningful accuracy gains from Opus 4.7's
            # adaptive thinking. VR (passage comprehension) and SJT (rubric
            # alignment) are mostly single-step inference and don't
            # benefit enough to justify the extra ~$0.01/call.
            use_thinking = section in {"DM", "QR"}
            with trace(
                "generate", section=section, model=self.llm,
                thinking=use_thinking,
            ) as gt:
                parsed, gen_usage = generate_structured(
                    system_blocks=system_blocks, user=user,
                    model=self.llm, output_schema=model_cls,
                    on_delta=on_delta, max_tokens=8000,
                    thinking=use_thinking,
                )
                gt.update(gen_usage)
            data = parsed.model_dump()
            data["section"] = section

            # Subtype drift check — fast, deterministic, no API call.
            subtype_drift = _detect_subtype_drift(section, data, subtype)

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
                "subtype_drift": subtype_drift,
            })

            # Dedup detection against KB.
            dup_warning = None
            if qvec is not None:
                try:
                    gen_text = embed_text_for(data)
                    # Dedup compares the freshly-generated doc against KB
                    # docs (also embedded as documents). Use the document
                    # input_type so both sides live in the same embedding
                    # space — Voyage is asymmetric, so a query-space vector
                    # would compare incorrectly against doc-space vectors.
                    gen_vec  = embed_doc(gen_text, self.emb)
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

        # Submit the async verify worker to the engine's executor AFTER the
        # trace closes so rag_generate's elapsed_ms reflects the user-visible
        # wait. The worker has its own `verify_async` trace span. Track the
        # future so `cancel_pending_verifies()` and `shutdown()` can manage it.
        #
        # The `_cancelled` check below is racey by design: a UI shutdown can
        # land between the check and submit(), so submit() can raise
        # RuntimeError("cannot schedule new futures after shutdown"). We catch
        # it explicitly rather than crashing the generate() caller — the
        # verify is a fire-and-forget enhancement, not a correctness
        # requirement, so silently dropping it on shutdown is correct.
        if async_verify and not self._cancelled:
            try:
                fut = self._verify_executor.submit(
                    self._async_verify_worker,
                    row_id, section, data, gen_usage, target,
                    verdict_dict, on_verify_complete,
                )
            except RuntimeError as err:
                # Executor was shut down between our check and submit.
                logger.info(
                    "async verify skipped: executor shut down (row_id=%s): %s",
                    row_id, err,
                )
            else:
                with self._verify_lock:
                    self._verify_futures.add(fut)
                # Done-callback removes the future from the active set whether
                # it completed normally, raised, or was cancelled.
                def _release(f: Future, _self=self) -> None:
                    with _self._verify_lock:
                        _self._verify_futures.discard(f)
                fut.add_done_callback(_release)

        return {
            "data":          data,
            "retrieved":     retrieved,
            "usage":         total_usage,
            "verdict":       verdict_dict,
            "coverage":      coverage_dict,
            "difficulty":    cal,
            "dup_warning":   dup_warning,
            "row_id":        row_id,
            "subtype_drift": subtype_drift,
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
        # Section-specific primary judge: SJT uses Sonnet because rubric
        # scoring is too subjective for Haiku; other sections stick with
        # the cheap Haiku default.
        primary_judge = PRIMARY_JUDGE_BY_SECTION.get(section, DEFAULT_VERIFY_LLM)

        if self.multi_judge:
            if on_progress: on_progress("Multi-judge jury verifying…")
            # Promote primary to first slot; tie-breaker is Sonnet (or, for
            # SJT where Sonnet is already primary, fall through to Opus).
            tiebreaker = (
                self.llm if primary_judge == DEFAULT_JUDGE2_LLM else DEFAULT_JUDGE2_LLM
            )
            judges = [primary_judge, tiebreaker, self.llm]
            # De-dup in case the user has llm=Sonnet and section=SJT —
            # otherwise we'd run the same model twice.
            seen: set[str] = set()
            judges = [j for j in judges if not (j in seen or seen.add(j))]
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

        if on_progress: on_progress(f"Verifying answers ({primary_judge.split('-')[1].title()})…")
        try:
            v, vu = llm_judge(section, data, primary_judge)
            verdict_dict = {"mode": "single",
                              "judge": primary_judge,
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

    # ── Lifecycle: cancellation + shutdown ───────────────────────────────────

    def cancel_pending_verifies(self) -> int:
        """Best-effort cancellation of queued verify tasks.

        Only futures that haven't started executing yet can be cancelled
        (in-flight Claude calls run to completion). Returns the count of
        successfully cancelled futures.

        Use case: the user kicks off a bulk run and then closes the
        dashboard. The UI calls this so queued verifies for already-saved
        rows don't waste API spend on results no one will see.
        """
        cancelled = 0
        with self._verify_lock:
            futures = list(self._verify_futures)
        for f in futures:
            if f.cancel():
                cancelled += 1
        return cancelled

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """Tear down the async verify executor cleanly.

        Args:
            wait: If True, block until all currently-running verifies finish
                  (default — preserves their results in the DB). If False,
                  return immediately and let in-flight tasks finish in the
                  background.
            cancel_pending: If True, also cancel queued (not-yet-started)
                  verifies before shutting down.

        After shutdown, no new verifies can be submitted; `generate()` will
        skip the verify spawn. Idempotent.
        """
        self._cancelled = True
        if cancel_pending:
            self.cancel_pending_verifies()
        # Python 3.9+ accepts cancel_futures; older fallback below.
        try:
            self._verify_executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        except TypeError:
            self._verify_executor.shutdown(wait=wait)
