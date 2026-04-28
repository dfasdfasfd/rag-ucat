"""
Structured chain-of-thought prompt engineering with style extraction,
section-specific instructions, difficulty targeting, retry prompts,
and prompt versioning for A/B tracking.
"""

import hashlib
import json
import re

from src.config import (
    SECTIONS,
    SECTION_SCHEMAS,
    SECTION_GEN_PARAMS,
    MINIGAME_KIND_DESCRIPTIONS,
    MINIGAME_HINT_REQUIREMENTS,
)


# ─── Prompt Versioning ───────────────────────────────────────────────────────

def prompt_version(system_template: str) -> str:
    """SHA256 hash of prompt template for A/B tracking."""
    return hashlib.sha256(system_template.encode()).hexdigest()[:12]


# ─── Style Extraction ────────────────────────────────────────────────────────

def extract_style_guide(retrieved_docs: list, section: str) -> str:
    """
    Analyze retrieved documents to extract patterns:
    - Average passage/stimulus length
    - Vocabulary complexity
    - Question structure patterns
    - Explanation detail level
    """
    if not retrieved_docs:
        return "No reference examples available."

    stats = {
        "passage_lengths": [],
        "question_counts": [],
        "explanation_lengths": [],
        "has_explanations": 0,
        "total_docs": len(retrieved_docs),
    }

    for doc in retrieved_docs:
        data = doc.get("data", {})

        # Passage / stimulus length
        for key in ("passage", "stimulus", "set_a_description"):
            if key in data:
                words = len(str(data[key]).split())
                stats["passage_lengths"].append(words)

        # Question analysis
        questions = data.get("questions", [])
        stats["question_counts"].append(len(questions))

        for q in questions:
            exp = q.get("explanation", "")
            if exp:
                stats["has_explanations"] += 1
                stats["explanation_lengths"].append(len(exp.split()))

    lines = []

    if stats["passage_lengths"]:
        avg_len = sum(stats["passage_lengths"]) // len(stats["passage_lengths"])
        lines.append(f"- Average passage/stimulus length: ~{avg_len} words")

    if stats["question_counts"]:
        avg_q = sum(stats["question_counts"]) // len(stats["question_counts"])
        lines.append(f"- Average questions per set: {avg_q}")

    if stats["explanation_lengths"]:
        avg_exp = sum(stats["explanation_lengths"]) // len(stats["explanation_lengths"])
        has_pct = (stats["has_explanations"] * 100) // max(1, sum(stats["question_counts"]))
        lines.append(f"- Explanations present: {has_pct}% of questions, avg ~{avg_exp} words")

    if section == "VR" and stats["passage_lengths"]:
        lines.append("- Passages should be academic in tone, information-dense")
    elif section == "QR":
        lines.append("- Stimulus should contain enough data for multi-step calculations")
        lines.append("- Explanations must show step-by-step working")
    elif section == "DM":
        lines.append("- Each question must be self-contained with clear logical structure")
    elif section == "SJT":
        lines.append("- Scenarios should be realistic professional situations with nuanced ethical considerations")

    return "\n".join(lines) if lines else "Follow the format of the reference examples."


# ─── Section-Specific Instructions ───────────────────────────────────────────

SECTION_INSTRUCTIONS = {
    "VR": (
        "- Write an original passage (200-300 words) on a DIFFERENT topic than the examples\n"
        "- Ensure each question is answerable ONLY from the passage\n"
        "- For True/False/Can't Tell: 'Can't Tell' means the passage doesn't provide enough info\n"
        "- Include a mix of explicit and inferential questions\n"
        "- The passage should be academic in tone with nuanced vocabulary"
    ),
    "DM": (
        "- Create exactly 5 standalone questions covering DIFFERENT reasoning types\n"
        "- Syllogisms: premises must be logically sound, conclusions testable\n"
        "- Probability: state values clearly, answer should be mathematically verifiable\n"
        "- Arguments: present a clear proposition with varying argument strengths\n"
        "- Venn: set relationships must be unambiguous\n"
        "- Each question must have exactly one correct answer"
    ),
    "QR": (
        "- Create a data table/chart with realistic numbers\n"
        "- Questions should require multi-step calculations\n"
        "- Show ALL working in each explanation (step-by-step)\n"
        "- Verify each numerical answer is correct before outputting\n"
        "- Options should be close enough to require precise calculation\n"
        "- Include different calculation types: percentages, ratios, averages, comparisons"
    ),
    "SJT": (
        "- Write a realistic professional scenario (80-150 words) involving an interpersonal or ethical dilemma\n"
        "- Scenarios should reflect situations a medical/healthcare professional might encounter\n"
        "- Each question should ask how appropriate a specific action is, or what the best response would be\n"
        "- Options must be rankable from most to least appropriate\n"
        "- Avoid obviously correct or obviously wrong answers — good SJT questions have nuance\n"
        "- Cover themes: teamwork, patient care, ethics, conflict resolution, professional integrity"
    ),
}


def build_minigame_instructions(section: str) -> str:
    """
    Inject the section's valid `minigame_kind` catalogue into the prompt so the
    LLM tags each question with the correct downstream minigame target. The
    Pocket UCAT app uses these tags to route questions into specific minigame
    formats (TFC, main-idea finder, paraphrase match, etc.).

    For kinds that need richer structured data (syllogism premises, venn items,
    table headers, role actions, etc.), also injects the `minigame_hints`
    schema specific to that kind.
    """
    schema = SECTION_SCHEMAS.get(section, {})
    kinds = schema.get("valid_minigame_kinds", [])
    if not kinds:
        return ""

    lines = [
        "## Minigame Tagging (REQUIRED)",
        "Add a `minigame_kind` field to EVERY question. This tags the question",
        f"for downstream rendering in the Pocket UCAT app. Allowed values for {section}:",
    ]
    for kind in kinds:
        desc = MINIGAME_KIND_DESCRIPTIONS.get(kind, "")
        lines.append(f"  - \"{kind}\" — {desc}" if desc else f"  - \"{kind}\"")
    lines.append("")
    lines.append("Choose the kind whose description best matches what the question is testing.")
    if section == "VR":
        lines.append(
            "For VR specifically: type:'tf' questions ALWAYS use minigame_kind:'tfc'. "
            "type:'mc' questions pick from main-idea, paraphrase, tone-purpose, or inference."
        )
    if section == "DM":
        lines.append(
            "For DM: keep `type` (syllogism/logical/venn/probability/argument) AND add "
            "`minigame_kind`. The mapping is: syllogism→syllogism, logical→logic-grid, "
            "venn→venn, probability→probability, argument→argument-strength."
        )

    # Structured hint requirements for kinds that need them
    hint_kinds = [k for k in kinds if k in MINIGAME_HINT_REQUIREMENTS]
    if hint_kinds:
        lines.append("")
        lines.append("## Structured Minigame Hints (REQUIRED for these kinds)")
        lines.append(
            "When using one of the kinds below, ALSO add a `minigame_hints` "
            "object with the structured fields shown. These hints unlock richer "
            "rendering in the app — without them the question falls back to a "
            "generic MCQ presentation."
        )
        lines.append("")
        for kind in hint_kinds:
            req = MINIGAME_HINT_REQUIREMENTS[kind]
            lines.append(f"### {kind}")
            lines.append(req["description"])
            lines.append(f"Example: `{req['shape']}`")
            lines.append("")

    return "\n".join(lines)


# ─── Difficulty Descriptions ─────────────────────────────────────────────────

def difficulty_desc(level: int = None) -> str:
    """Map difficulty 1-10 to a description for the prompt."""
    if level is None:
        return "Match the difficulty level of the reference examples"
    if level <= 3:
        return f"Easy (difficulty {level}/10): straightforward, single-step reasoning"
    elif level <= 6:
        return f"Medium (difficulty {level}/10): requires careful reading and multi-step thinking"
    else:
        return f"Hard (difficulty {level}/10): complex, multi-layered reasoning with subtle distinctions"


# ─── Main Prompt Builder ─────────────────────────────────────────────────────

class PromptBuilder:
    """Builds structured generation and retry prompts."""

    def build(self, section: str, retrieved_docs: list,
              hint: str = "", difficulty: int = None) -> tuple:
        """
        Build system + user prompts for question generation.
        Returns: (system_prompt, user_prompt, version_hash)
        """
        schema = SECTION_SCHEMAS[section]
        style_guide = extract_style_guide(retrieved_docs, section)
        section_instr = SECTION_INSTRUCTIONS.get(section, "")
        minigame_instr = build_minigame_instructions(section)
        diff_desc = difficulty_desc(difficulty)

        # Format retrieved docs as context
        context_blocks = []
        for i, doc in enumerate(retrieved_docs):
            score = doc.get("_rrf_score", 0)
            data_json = json.dumps(doc.get("data", {}), indent=2)
            context_blocks.append(
                f"--- REFERENCE EXAMPLE {i+1} (relevance: {score:.4f}) ---\n{data_json}"
            )
        context = "\n\n".join(context_blocks) if context_blocks else "(No reference examples available)"

        system = f"""You are an expert UCAT {SECTIONS[section]} question author.

## Your Task
Generate a completely new, original UCAT {SECTIONS[section]} question set.

## Style Guide (extracted from {len(retrieved_docs)} reference examples)
{style_guide}

## Section-Specific Requirements
{section_instr}

{minigame_instr}

## Difficulty
{diff_desc}

## Output Format
Return ONLY valid JSON matching this exact schema:
{schema['shape']}

Section description: {schema['desc']}

## Reference Examples from Knowledge Base
The following examples define the authoritative format, difficulty, and style you MUST follow.
Your output must mirror their JSON structure exactly.
Do NOT copy content from these examples — create entirely original content.

{context}

## Think step by step:
1. Study the reference examples to understand the expected format and quality
2. Choose a DIFFERENT topic/theme than the examples
3. Create original content (passage/stimulus/descriptions)
4. Write questions that test the same cognitive skills as the examples
5. Verify each answer is unambiguously correct
6. Ensure distractors are plausible but definitively wrong
7. Add clear explanations for each answer
8. Output the complete JSON"""

        user = (
            f"Generate a completely NEW and UNIQUE UCAT {SECTIONS[section]} question set. "
            + (f"Focus on the topic: {hint.strip()}. " if hint.strip() else "")
            + "All content, numbers, scenarios, and passages must be entirely original. "
            "Follow the format of the reference documents precisely. "
            "Return ONLY valid JSON."
        )

        version = prompt_version(system)
        return system, user, version

    def build_retry(self, original_system: str, original_user: str,
                    broken_output: str, errors: list) -> tuple:
        """
        Build retry prompt with original context + broken output + specific errors.
        Gives the LLM everything it needs to fix exactly what went wrong.
        """
        error_list = "\n".join(f"  - {e}" for e in errors)

        system = (
            original_system + "\n\n"
            "## ⚠️ CORRECTION REQUIRED\n"
            "Your previous output had the following errors:\n"
            f"{error_list}\n\n"
            "Your previous (broken) output was:\n"
            f"```json\n{broken_output[:3000]}\n```\n\n"
            "Fix ONLY the listed errors. Do not change anything that was correct. "
            "Return the complete corrected JSON."
        )

        return system, original_user

    def build_dedup_retry(self, original_system: str, original_user: str,
                          context: str = "KB") -> tuple:
        """Build retry prompt when output is too similar to existing content."""
        user = (
            original_user + "\n\n"
            f"IMPORTANT: Your previous output was too similar to existing {context} content. "
            "Generate something substantially DIFFERENT — change the topic, scenario, "
            "or approach entirely while maintaining the same format."
        )
        return original_system, user

    def build_calibration_penalty(self, original_system: str,
                                  original_user: str) -> tuple:
        """Build retry prompt after calibration rejection."""
        user = (
            original_user + "\n\n"
            "IMPORTANT: The previous output was rejected during quality calibration. "
            "Generate a substantially different question set with higher quality. "
            "Pay extra attention to: answer correctness, distractor plausibility, "
            "and passage/stimulus quality."
        )
        return original_system, user
