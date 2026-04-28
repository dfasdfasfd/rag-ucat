"""Pydantic schemas for structured Claude outputs.

Three new wrinkles vs. the prototype:

1. **Structured visual specs.** AR panels, QR charts, and DM Venn diagrams
   are now machine-renderable specs (not Unicode descriptions). The
   ``rendering`` module turns them into PNGs at display time.

2. **Per-question IRT difficulty.** Claude predicts a logit on the 1.0-5.0
   Rasch scale alongside each question, plus a topic + scenario_type so the
   coverage analyzer can spot bias and gaps.

3. **Multi-judge verdicts.** The verifier reports per-question correctness,
   not just a global flag, so we can flag specific items rather than the
   whole set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

# ─── Shared building blocks ───────────────────────────────────────────────────

ScenarioType = Literal[
    "scientific", "humanities", "business", "medical",
    "everyday", "sport", "abstract", "social",
]


class CoverageTags(BaseModel):
    """Per-question metadata emitted by the model — drives bias/coverage detection."""
    topic: str                  # e.g. "ecology", "personal finance"
    scenario_type: ScenarioType
    contains_named_entities: bool = False
    cultural_context: Optional[str] = None  # e.g. "UK", "general", "non-Western"


# Path-C addition: coarse "how close are the wrong answers" signal that
# combines with `solve_steps` to give downstream calibration a stable
# basis for difficulty inference. See enricher's COMMON FIELDS reminder.
DistractorProximity = Literal["near", "moderate", "far"]


class OptionItem(BaseModel):
    """One answer choice. Modelled as a struct (not a Dict[str, str]) because
    Anthropic strict mode rejects ``additionalProperties: <schema>`` map types.
    Question.options serializes this list back to ``{label: text}`` for
    downstream code (format, calibration, verification, db)."""
    model_config = ConfigDict(extra="forbid")
    label: str = Field(description="Option label shown to the student, e.g. 'A', 'B', 'True', 'Set A'.")
    text:  str = Field(description="Option text.")


class Question(BaseModel):
    number: int
    text: str
    type: Optional[str] = None  # DM subtype: syllogism / logical / probability / argument / venn
    minigame_kind: Optional[str] = Field(
        default=None,
        description=(
            "Pocket UCAT routing tag. For VR subtype targeting: one of "
            "'tfc', 'main-idea', 'paraphrase', 'tone-purpose', 'inference', "
            "'vocabulary', 'application'. Optional — legacy rows and mixed-mode runs may leave this null."
        ),
    )
    options: List[OptionItem] = Field(
        min_length=2, max_length=8,
        description=(
            "Answer choices as an ordered list of {label, text} objects. "
            "VR/QR multiple-choice: 4 items with labels A,B,C,D. "
            "VR True/False/Can't Tell: 3 items with labels exactly 'True', 'False', 'Can't Tell'. "
            "DM: 5 items with labels A,B,C,D,E. AR test items: 3 items with labels 'Set A', 'Set B', 'Neither'. "
            "Never emit an empty list."
        ),
    )
    answer: str
    explanation: str
    difficulty: float = Field(ge=1.0, le=5.0,
                              description="IRT logits on the 1.0-5.0 scale.")
    coverage: CoverageTags

    # Path-C generic enrichment fields (audit-3 M1 + M2). Replace Claude's
    # noisy 1.0-5.0 difficulty guess by giving the calibrator less-noisy
    # raw inputs (`solve_steps` + `distractor_proximity`); downstream IRT
    # and judge-blending stay backward-compatible while the new fields
    # arrive. `common_mistake` carries error-pattern hints lifted from
    # the explanation panel — feeds distractor generation.
    solve_steps: Optional[int] = Field(
        default=None, ge=0, le=20,
        description=(
            "Coarse complexity proxy. QR: arithmetic-op count. DM: inference-hop count. "
            "VR: distinct passage references the candidate must reconcile. SJT: usually 1-2."
        ),
    )
    distractor_proximity: Optional[DistractorProximity] = Field(
        default=None,
        description=(
            "How close the wrong answers cluster to the right one. "
            "'near' = within ~10% numerically or differs by one logical clause. "
            "'far' = obviously wrong by inspection. 'moderate' = anything between."
        ),
    )
    common_mistake: Optional[str] = Field(
        default=None,
        description=(
            "Typical error pattern lifted from the explanation panel "
            "(e.g. 'candidates often divide instead of multiplying'). "
            "Pure gold for distractor generation downstream."
        ),
    )

    @field_validator("options", mode="before")
    @classmethod
    def _accept_legacy_dict(cls, v: Any) -> Any:
        """Older KB rows (samples.py, crawler_import.py, pre-fix DB) store options
        as ``{label: text}``. Accept that shape so re-validation still works."""
        if isinstance(v, dict):
            return [{"label": str(k), "text": str(t)} for k, t in v.items()]
        return v

    @field_serializer("options")
    def _serialize_options_as_dict(self, options: List["OptionItem"]) -> Dict[str, str]:
        """Downstream code reads options as a dict. Flatten the list of pairs
        back to that shape on dump so format/calibration/verification/db all
        keep working unchanged."""
        return {o.label: o.text for o in options}

# ─── Visual specs ─────────────────────────────────────────────────────────────

# AR — abstract reasoning shape sets.

ARShapeKind  = Literal["square", "circle", "triangle", "diamond", "star", "pentagon", "hexagon", "cross", "arrow"]
ARShapeColor = Literal["black", "white", "grey"]
ARShapeSize  = Literal["small", "medium", "large"]


class ARShape(BaseModel):
    kind: ARShapeKind
    color: ARShapeColor = "black"
    size: ARShapeSize = "medium"
    rotation_deg: int = Field(default=0, ge=0, le=359)
    # Position is auto-laid-out by the renderer; the model doesn't pick coords.


class ARPanel(BaseModel):
    label: str = ""           # optional human label, e.g. "Panel 1"
    shapes: List[ARShape]


class ARSet(BaseModel):
    section: Literal["AR"]
    set_a_panels: List[ARPanel] = Field(min_length=6, max_length=6)
    set_a_rule: str
    set_b_panels: List[ARPanel] = Field(min_length=6, max_length=6)
    set_b_rule: str
    test_panels: List[ARPanel] = Field(min_length=5, max_length=5,
                                         description="Test shapes shown to the student — one per question.")
    questions: List[Question] = Field(min_length=5, max_length=5)


# QR — quantitative reasoning data stimulus.

# `matrix` added in Path-C (audit-3 H3) to support transition tables /
# cross-tabs (voter transitions, contingency tables) where both axes
# carry meaningful categories — collapsing them into categories+series
# loses dual-axis semantics.
QRChartType = Literal["table", "bar", "line", "stacked_bar", "pie", "matrix"]


class QRSeries(BaseModel):
    """One data series for bar/line/stacked charts."""
    name: str
    values: List[float]


class QRTableColumn(BaseModel):
    """One column of a QR table. Modelled as a struct (not a dict) for the same
    reason as OptionItem — Anthropic strict mode rejects map-typed
    ``additionalProperties``. ``QRChart.rows`` flattens the list back to
    ``{name: values}`` on dump for downstream rendering / verification."""
    model_config = ConfigDict(extra="forbid")
    name:   str = Field(description="Column header.")
    values: List[str] = Field(
        description=(
            "Cell values down the column as strings, parallel to QRChart.categories "
            "(one per row). Numeric cells should be stringified (e.g. '12.5', '150'); "
            "non-numeric cells like 'N/A' or '12.5%' are also allowed."
        ),
    )


class QRMatrix(BaseModel):
    """A 2-D cross-tabulation with meaningful row AND column axes.

    Used for transition tables (e.g. voter transitions April→September),
    contingency tables, dose × outcome cross-tabs. The 1-D `categories +
    series + rows` representation flattens dual-axis semantics; this
    struct preserves them so the verifier and renderer can reason about
    "row R, column C" cells directly.
    """
    model_config = ConfigDict(extra="forbid")
    row_axis_label: str
    col_axis_label: str
    row_categories: List[str] = Field(min_length=2)
    col_categories: List[str] = Field(min_length=2)
    cells: List[List[str]] = Field(
        description=(
            "2-D array indexed [row_i][col_j]. Cell values as strings — "
            "numerics should be stringified (e.g. '12.5')."
        ),
    )
    row_totals: Optional[List[str]] = None
    col_totals: Optional[List[str]] = None


class QRChart(BaseModel):
    type: QRChartType
    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    categories: List[str] = Field(default_factory=list,
                                    description="X-axis categories or pie segment labels.")
    series: List[QRSeries] = Field(default_factory=list,
                                     description="Empty for tables/pies — use 'rows' for tables, 'segments' for pies.")
    rows: Optional[List[QRTableColumn]] = Field(
        default=None,
        description=(
            "Table form ONLY (omit for bar/line/stacked_bar/pie). "
            "List of {name, values} columns; values length must match categories length."
        ),
    )
    matrix: Optional[QRMatrix] = Field(
        default=None,
        description=(
            "Cross-tabulation form ONLY (matrix-typed charts). "
            "Use when the figure is a 2-D grid with meaningful labels on both axes."
        ),
    )
    units: Optional[str] = None  # e.g. "£000s", "%"
    note: Optional[str] = None   # optional caption
    # Path-C: source attribution / asterisk-marked footnotes lifted from
    # the chart caption area. Often-overlooked context — e.g.
    # "*excludes weekends", "Source: ONS 2022".
    footnotes: List[str] = Field(default_factory=list)

    @field_validator("rows", mode="before")
    @classmethod
    def _accept_legacy_rows_dict(cls, v: Any) -> Any:
        """Older KB rows store rows as ``{col_name: [values]}``. Accept that
        shape so re-validation still works."""
        if isinstance(v, dict):
            return [{"name": str(k), "values": list(vs or [])} for k, vs in v.items()]
        return v

    @field_serializer("rows")
    def _serialize_rows_as_dict(self, rows: Optional[List["QRTableColumn"]]
                                ) -> Optional[Dict[str, List[Union[str, float]]]]:
        """Downstream code (format, verification.symbolic_qr_check, rendering)
        reads ``rows`` as ``{col_name: [values]}``. Flatten on dump so that
        contract is preserved."""
        if rows is None:
            return None
        return {col.name: list(col.values) for col in rows}


# QR-specific skill taxonomy (audit-3 H1) — primary axis for retrieval
# diversification. Free-text `coverage.topic` stays as secondary annotation.
QRSkillTag = Literal[
    "ratio-percent",
    "mean-aggregate",
    "drug-dosage",
    "unit-conversion",
    "geometry-area",
    "prob-frequency",
    "compound-growth",
    "data-readout",
    "speed-distance-time",
    "scaling-extrapolation",
    "other",
]


class QRQuestion(Question):
    """QR extends Question with a closed skill_tag enum so retrieval can
    diversify on a stable axis (free-text `topic` cardinality explodes
    and rarely matches). Optional for back-compat with pre-Path-C rows."""
    skill_tag: Optional[QRSkillTag] = None


class QRSet(BaseModel):
    section: Literal["QR"]
    stimulus: QRChart
    questions: List[QRQuestion] = Field(min_length=4, max_length=4)


# DM — Decision Making with optional venn structure.

class VennSet(BaseModel):
    label: str
    members: List[str]


class DMVenn(BaseModel):
    """A 2- or 3-circle Venn drawn for venn-type DM questions."""
    sets: List[VennSet] = Field(min_length=2, max_length=3)
    universe_label: Optional[str] = None


# DM-specific skill taxonomy (audit-3 H1) — primary axis for retrieval.
DMSkillTag = Literal[
    "syllogism-conditional",
    "syllogism-quantifier",
    "venn-2set",
    "venn-3set",
    "prob-conditional",
    "prob-tree",
    "argument-strength",
    "recognise-assumption",
    "logical-puzzle",
    "interpreting-information",
    "other",
]


class DMQuestion(Question):
    """DM extends Question with optional structured venn data and a
    skill_tag for retrieval diversification."""
    venn: Optional[DMVenn] = None
    skill_tag: Optional[DMSkillTag] = None
    # For syllogism-typed questions in Pearson's per-conclusion Yes/No
    # layout: parallel array to `options` indicating each conclusion's
    # ground-truth Yes/No verdict (audit-3 H2). Surfaces the full Yes/No
    # signal that the answer letter alone discards (4 of 5 verdicts in
    # the typical 5-conclusion layout).
    conclusion_validity: Optional[List[Literal["yes", "no"]]] = Field(
        default=None,
        max_length=8,
    )


class DMSet(BaseModel):
    section: Literal["DM"]
    questions: List[DMQuestion] = Field(min_length=5, max_length=5)


# VR — pure text passage; no visuals.

# UCAT VR question kinds — drives subtype-locked retrieval and bulk
# variety steering. The role block uses these names; the trainer's
# generated VR docs include them. Crawler-imported docs may not — they
# fall back to `Question.type` (tf/mc) which is a structurally-narrower
# axis than the kind taxonomy.
VRMinigameKind = Literal[
    "tfc",          # True / False / Can't Tell statements
    "main-idea",    # central thesis / best title / overall conclusion
    "paraphrase",   # which option restates a quoted passage fragment
    "tone-purpose", # author tone, attitude, rhetorical purpose
    "inference",    # what can be concluded from the passage
    "vocabulary",   # word substitution / lexical knowledge (audit-3 M4)
    "application",  # transfer the argument to a new scenario (audit-3 M4)
    "other",        # explicit fallback for unclassifiable items
]


class VRQuestion(Question):
    """VR extends Question with an optional minigame_kind tag — a narrower
    classification than the inherited `type: tf|mc`. Generation always
    emits this; legacy crawler-imported docs may omit it (they have only
    the broader `type` field). Used by retrieval subtype filtering and
    coverage diversification."""
    minigame_kind: Optional[VRMinigameKind] = None


class VRSet(BaseModel):
    section: Literal["VR"]
    passage: str
    questions: List[VRQuestion] = Field(min_length=4, max_length=4)


# SJT — workplace/clinical scenario; pure text, no visuals.

SJTQuestionType = Literal["appropriateness", "importance"]

# UCAT SJT scenarios fall into recognisable situation families. Tracking
# this on the set lets `pick_diversification` steer bulk runs across
# situation types — without it, a 10-set bulk run can produce 9 medical-
# ethics scenarios and 0 boundary-management ones.
#
# Path-C expansion (audit-3 M3): real captures show 4 categories missing
# from the original 4-value enum. Now 8 values.
SJTSituationType = Literal[
    "medical_ethics",            # informed consent, end-of-life, confidentiality
    "team_conflict",             # disagreement with colleague / senior
    "boundary_management",       # scope of practice, dual relationships, gifts
    "professional_communication", # handover, error reporting, breaking bad news
    "personal_wellbeing",        # colleague struggling, candidate's own welfare
    "student_supervision",       # teaching, supervising, mentoring junior staff
    "consent_capacity",          # capacity assessment, refusal of treatment
    "resource_allocation",       # triage, rationing, time-critical prioritisation
]

# SJT-specific skill taxonomy (audit-3 H1).
SJTSkillTag = Literal[
    "consent-capacity",
    "confidentiality",
    "error-disclosure",
    "team-conflict",
    "boundary-management",
    "wellbeing",
    "priorities",
    "patient-safety",
    "honesty-integrity",
    "resource-stewardship",
    "other",
]


class SJTQuestion(Question):
    """SJT extends Question with an optional sub-type tag identifying whether
    the question is appropriateness-style (rate a candidate action) or
    importance-style (rate a consideration). Claude classifies this during
    enrichment when the question wording makes it unambiguous."""
    type: Optional[SJTQuestionType] = None  # narrows the inherited str type
    skill_tag: Optional[SJTSkillTag] = None
    # Position of the marked answer on the canonical Likert ordering
    # (1=top "Very appropriate"/"Very important", 4=opposite end).
    # Lets verification compute partial-credit (adjacent label = partial)
    # without re-deriving from prose (audit-3 M5).
    likert_rank: Optional[int] = Field(default=None, ge=1, le=4)


class SJTSet(BaseModel):
    section: Literal["SJT"]
    scenario: str = Field(min_length=1)
    # Situation-type tag for diversification tracking. Optional for back-
    # compat with crawler-imported SJT sets that pre-date this field;
    # generation produced post-fix should always set it.
    situation_type: Optional[SJTSituationType] = None
    questions: List[SJTQuestion] = Field(min_length=4, max_length=4)


SECTION_MODELS = {"VR": VRSet, "DM": DMSet, "QR": QRSet, "AR": ARSet, "SJT": SJTSet}

# ─── Verifier outputs ────────────────────────────────────────────────────────

class QuestionVerdict(BaseModel):
    number: int
    correct: bool
    reasoning: str = ""


class Verdict(BaseModel):
    """A single judge's verdict on the whole set."""
    per_question: List[QuestionVerdict] = Field(default_factory=list)
    overall_correct: bool
    confidence: Literal["low", "medium", "high"]
    notes: List[str] = Field(default_factory=list)


class JuryVerdict(BaseModel):
    """Aggregated multi-judge result."""
    judges: List[str]
    individual: List[Verdict]
    overall_correct: bool        # majority vote
    unanimous: bool
    flagged_questions: List[int] # numbers where any judge said incorrect
