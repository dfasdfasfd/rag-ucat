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
            "'tfc', 'main-idea', 'paraphrase', 'tone-purpose', 'inference'. "
            "Optional — legacy rows and mixed-mode runs may leave this null."
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

QRChartType = Literal["table", "bar", "line", "stacked_bar", "pie"]


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
    values: List[Union[str, float]] = Field(
        description="Cell values down the column, parallel to QRChart.categories (one per row).",
    )


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
    units: Optional[str] = None  # e.g. "£000s", "%"
    note: Optional[str] = None   # optional caption

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


class QRSet(BaseModel):
    section: Literal["QR"]
    stimulus: QRChart
    questions: List[Question] = Field(min_length=4, max_length=4)


# DM — Decision Making with optional venn structure.

class VennSet(BaseModel):
    label: str
    members: List[str]


class DMVenn(BaseModel):
    """A 2- or 3-circle Venn drawn for venn-type DM questions."""
    sets: List[VennSet] = Field(min_length=2, max_length=3)
    universe_label: Optional[str] = None


class DMQuestion(Question):
    """DM extends Question with optional structured venn data."""
    venn: Optional[DMVenn] = None


class DMSet(BaseModel):
    section: Literal["DM"]
    questions: List[DMQuestion] = Field(min_length=5, max_length=5)


# VR — pure text passage; no visuals.

class VRSet(BaseModel):
    section: Literal["VR"]
    passage: str
    questions: List[Question] = Field(min_length=4, max_length=4)


SECTION_MODELS = {"VR": VRSet, "DM": DMSet, "QR": QRSet, "AR": ARSet}

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
