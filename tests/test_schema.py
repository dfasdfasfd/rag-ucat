"""Tests for ucat.llm.pydantic_to_strict_schema and the Question.options shape.

Two regressions guarded against:

1. **Empty options dict.** ``Question.options`` was originally ``Dict[str, str]``
   with ``min_length=2``. Pydantic emits dicts with
   ``additionalProperties: <schema>`` — but Anthropic strict mode REJECTS that
   form ("``additionalProperties: object`` is not supported. Please set
   ``additionalProperties`` to false"). The previous "fix" of clobbering it to
   ``false`` produced a schema saying "no properties allowed", so Claude
   correctly emitted ``{}`` and Pydantic's ``min_length=2`` rejected.

   Resolution: ``options`` is now ``List[OptionItem]`` on the schema, with a
   serializer that flattens back to ``Dict[str, str]`` for downstream code.

2. **Strict-mode incompatibility.** No part of the strictified schema may
   contain ``additionalProperties: <dict>``; Anthropic rejects every such node.

Run directly:
    ./venv/bin/python tests/test_schema.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucat.llm import pydantic_to_strict_schema
from ucat.models import ARSet, DMSet, QRChart, Question, QRSet, SJTSet, VRSet


def _walk(schema: Any) -> Iterator[dict]:
    if isinstance(schema, dict):
        yield schema
        for v in schema.values():
            yield from _walk(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _walk(v)


def _options_schema(strict: dict) -> dict:
    return strict["properties"]["questions"]["items"]["properties"]["options"]


def test_no_map_typed_additionalProperties_anywhere():
    """Anthropic strict mode rejects every ``additionalProperties: <schema>``
    node. Walk every section schema and confirm there are none."""
    for cls in (VRSet, DMSet, QRSet, ARSet, SJTSet):
        strict = pydantic_to_strict_schema(cls)
        for node in _walk(strict):
            ap = node.get("additionalProperties")
            assert ap is None or ap is False, (
                f"{cls.__name__}: found map-typed additionalProperties={ap!r} "
                f"in node keys {sorted(node.keys())}"
            )


def test_options_is_array_of_label_text_objects():
    """The schema sent to Claude must describe options as a list of
    ``{label, text}`` objects so the API accepts it AND Claude has a clear
    structure to emit."""
    strict = pydantic_to_strict_schema(VRSet)
    opts = _options_schema(strict)
    assert opts.get("type") == "array", (
        f"options.type must be 'array', got {opts!r}"
    )
    item = opts.get("items") or {}
    assert item.get("type") == "object", (
        f"options.items.type must be 'object', got {item!r}"
    )
    item_props = item.get("properties") or {}
    assert "label" in item_props and "text" in item_props, (
        f"options.items must have label+text properties, got {sorted(item_props)}"
    )
    required = set(item.get("required") or [])
    assert {"label", "text"}.issubset(required), (
        f"label/text must be required on each option item, got required={required}"
    )


def test_question_serializes_options_as_dict_for_downstream():
    """Downstream code (format, calibration, verification, db) reads
    ``q['options']`` as a ``Dict[str, str]``. The Pydantic serializer must
    flatten the list-of-pairs back to that shape so that contract is preserved."""
    q = Question.model_validate({
        "number": 1,
        "text": "Stem.",
        "options": [
            {"label": "A", "text": "alpha"},
            {"label": "B", "text": "beta"},
        ],
        "answer": "A",
        "explanation": "because",
        "difficulty": 2.5,
        "coverage": {"topic": "x", "scenario_type": "scientific"},
    })
    dumped = q.model_dump()
    assert isinstance(dumped["options"], dict), (
        f"options must dump as dict for downstream code, got {type(dumped['options']).__name__}"
    )
    assert dumped["options"] == {"A": "alpha", "B": "beta"}, dumped["options"]


def test_question_validator_accepts_legacy_dict_shape():
    """Existing KB documents (samples.py, crawler_import.py) store options as
    a dict. Re-validating those rows must still work."""
    q = Question.model_validate({
        "number": 1,
        "text": "Stem.",
        "options": {"A": "alpha", "B": "beta"},
        "answer": "A",
        "explanation": "because",
        "difficulty": 2.5,
        "coverage": {"topic": "x", "scenario_type": "scientific"},
    })
    # Round-trips back to dict shape via the serializer.
    assert q.model_dump()["options"] == {"A": "alpha", "B": "beta"}


def test_qrchart_rows_is_array_of_column_objects():
    """QRChart.rows had the same map-type bug as Question.options. The schema
    must now describe rows as a list of column objects (Anthropic-compatible),
    not an empty closed object that bars all keys."""
    strict = pydantic_to_strict_schema(QRChart)
    rows = strict["properties"]["rows"]
    branches = rows.get("anyOf", [])
    assert branches, f"expected anyOf for Optional rows, got {rows!r}"
    array_branch = next((b for b in branches if b.get("type") == "array"), None)
    assert array_branch is not None, (
        f"rows must have an array branch, got {branches!r}"
    )
    item = array_branch.get("items") or {}
    assert item.get("type") == "object", (
        f"rows.items.type must be 'object', got {item!r}"
    )
    item_props = item.get("properties") or {}
    assert "name" in item_props and "values" in item_props, (
        f"rows.items must have name+values, got {sorted(item_props)}"
    )


def test_qrchart_serializes_rows_as_dict_for_downstream():
    """format.py and verification.py read ``stim['rows']`` as
    ``{col_name: [values]}``. The serializer must keep emitting that shape."""
    chart = QRChart.model_validate({
        "type": "table",
        "title": "Sales",
        "categories": ["Q1", "Q2"],
        "rows": [
            {"name": "London", "values": [10.0, 12.0]},
            {"name": "Paris",  "values": [8.0, 11.5]},
        ],
    })
    dumped = chart.model_dump()
    assert isinstance(dumped["rows"], dict), (
        f"rows must dump as dict, got {type(dumped['rows']).__name__}"
    )
    assert dumped["rows"] == {"London": [10.0, 12.0], "Paris": [8.0, 11.5]}


def test_qrchart_validator_accepts_legacy_rows_dict():
    """Existing KB chart specs store rows as ``{col_name: [values]}``.
    Re-validation must keep working."""
    chart = QRChart.model_validate({
        "type": "table",
        "title": "Sales",
        "categories": ["Q1", "Q2"],
        "rows": {"London": [10.0, 12.0], "Paris": [8.0, 11.5]},
    })
    assert chart.model_dump()["rows"] == {"London": [10.0, 12.0], "Paris": [8.0, 11.5]}


def test_qrchart_rows_optional_none_round_trips():
    """rows is Optional — bar/line/pie charts have rows=None."""
    chart = QRChart.model_validate({
        "type": "bar",
        "title": "Sales",
        "categories": ["A", "B"],
        "series": [{"name": "y", "values": [1.0, 2.0]}],
    })
    assert chart.model_dump()["rows"] is None


def test_closed_objects_still_get_strict_treatment():
    """Pydantic models (closed objects) must still get
    ``additionalProperties: false`` and a full ``required`` list."""
    strict = pydantic_to_strict_schema(Question)
    assert strict.get("additionalProperties") is False
    required = set(strict.get("required") or [])
    for field in ("number", "text", "options", "answer", "explanation",
                  "difficulty", "coverage"):
        assert field in required, f"{field} missing from required: {required}"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed.")
