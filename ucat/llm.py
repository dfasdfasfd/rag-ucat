"""Anthropic Claude + Voyage AI client wrappers.

Handles client lifecycle, structured-output schema conversion, prompt-cache
breakpoint placement, and per-call usage/cost extraction.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from .config import MODEL_COSTS, DEFAULT_LLM, EMBED_BATCH_SIZE
from .telemetry import logger

try:
    import anthropic
    import voyageai
    _OK = True
    _ERR = None
except ImportError as e:
    _OK = False
    _ERR = str(e)


# ─── Singletons ───────────────────────────────────────────────────────────────

class LLMClient:
    _anth: Optional[Any] = None
    _vo:   Optional[Any] = None

    @classmethod
    def anthropic(cls):
        if not _OK:
            raise RuntimeError(f"anthropic/voyageai not installed: {_ERR}")
        if cls._anth is None:
            # 60s per-request timeout vs the SDK default of 600s — keeps a stalled
            # verify call from blocking the UI for ten minutes. max_retries=4 (vs
            # default 2) absorbs the bursty `overloaded_error` 529s we see at peak.
            cls._anth = anthropic.Anthropic(timeout=60.0, max_retries=4)
        return cls._anth

    @classmethod
    def voyage(cls):
        if not _OK:
            raise RuntimeError(f"anthropic/voyageai not installed: {_ERR}")
        if cls._vo is None:
            cls._vo = voyageai.Client()
        return cls._vo


# ─── Embeddings ───────────────────────────────────────────────────────────────

def embed_doc(text: str, model: str) -> List[float]:
    """Single-doc embedding (input_type='document')."""
    res = LLMClient.voyage().embed([text], model=model, input_type="document")
    return res.embeddings[0]


def embed_query(text: str, model: str) -> List[float]:
    """Query-side embedding (input_type='query'). Asymmetric retrieval."""
    res = LLMClient.voyage().embed([text], model=model, input_type="query")
    return res.embeddings[0]


def embed_batch(texts: List[str], model: str, input_type: str = "document") -> List[List[float]]:
    """Batched embedding — chunked at EMBED_BATCH_SIZE."""
    if not texts:
        return []
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[i:i + EMBED_BATCH_SIZE]
        res   = LLMClient.voyage().embed(chunk, model=model, input_type=input_type)
        out.extend(res.embeddings)
    return out


# ─── Schema conversion ───────────────────────────────────────────────────────

def pydantic_to_strict_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic v2 model to a JSON Schema acceptable by Claude's
    ``output_config.format`` — inlines $refs, forces additionalProperties:false,
    strips constraints the Anthropic API rejects.

    Anthropic's structured-output endpoint rejects:
      • ``minItems``/``maxItems`` other than 0 or 1
      • ``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum``/``multipleOf``
      • ``minLength``/``maxLength``/``pattern``
      • ``minProperties``/``maxProperties`` (emitted by Pydantic for ``min_length``
        on ``Dict`` fields, since dict length = property count)

    Pydantic still validates these after parsing, so we get the constraint
    enforcement client-side; we just can't push it into the JSON Schema."""
    schema = model_cls.model_json_schema()
    return _strictify(schema, schema.get("$defs", {}))


# Constraints the API rejects in `output_config.format.schema`.
# (Pydantic still enforces them on the parsed instance after the call.)
_UNSUPPORTED_KEYS = {
    "minItems", "maxItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern",
    "minProperties", "maxProperties",
}


def _strictify(schema: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return _strictify(defs[ref], defs)
    out = {k: v for k, v in schema.items()
           if k != "$defs" and k not in _UNSUPPORTED_KEYS}
    t = out.get("type")
    if t == "object":
        # Anthropic strict mode does not accept ``additionalProperties: <schema>``
        # (map types). Pydantic ``Dict[str, X]`` fields therefore have to be
        # modelled as list-of-pair objects on the schema side — see Question.options.
        props = out.get("properties", {})
        out["properties"] = {k: _strictify(v, defs) for k, v in props.items()}
        out["additionalProperties"] = False
        if "required" not in out:
            out["required"] = list(out["properties"].keys())
    if t == "array" and isinstance(out.get("items"), dict):
        out["items"] = _strictify(out["items"], defs)
    for key in ("anyOf", "allOf", "oneOf"):
        if key in out:
            out[key] = [_strictify(s, defs) for s in out[key]]
    return out


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_structured(
    *,
    system_blocks: List[Dict[str, Any]],
    user: str,
    model: str,
    output_schema: Type[BaseModel],
    on_delta: Optional[Callable[[str], None]] = None,
    max_tokens: int = 8000,
    thinking: bool = False,
) -> Tuple[BaseModel, Dict[str, Any]]:
    """
    Streamed Claude generation that returns a validated Pydantic instance.
    System is a list of blocks so the stable prefix gets cache_control while
    the volatile suffix doesn't.

    When `thinking=True`, adaptive thinking is enabled — Claude decides per
    request how much to think. Recommended for sections that require
    multi-step reasoning (DM venn/probability, QR multi-step calculations).
    Costs ~$0.005-0.015 extra per call but materially reduces calculation
    errors and logical-leap mistakes.
    """
    client = LLMClient.anthropic()
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": pydantic_to_strict_schema(output_schema),
            }
        },
    }
    if thinking:
        # Adaptive thinking — Opus 4.7 chooses budget per request. We omit
        # display="summarized" intentionally: streaming is to a UI text
        # buffer that doesn't render thinking content.
        kwargs["thinking"] = {"type": "adaptive"}

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                if on_delta:
                    on_delta(event.delta.text)
        msg = stream.get_final_message()

    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        data = json.loads(text)
        parsed = output_schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("Schema validation failed: %s", e)
        raise RuntimeError(f"Model output failed schema validation: {e}")

    return parsed, extract_usage(msg, model)


# ─── Usage / cost ────────────────────────────────────────────────────────────

def extract_usage(msg: Any, model: str) -> Dict[str, Any]:
    u = msg.usage
    in_t   = getattr(u, "input_tokens", 0) or 0
    out_t  = getattr(u, "output_tokens", 0) or 0
    cw     = getattr(u, "cache_creation_input_tokens", 0) or 0
    cr     = getattr(u, "cache_read_input_tokens", 0) or 0
    cost_table = MODEL_COSTS.get(model, MODEL_COSTS[DEFAULT_LLM])
    dollars = (
        in_t  * cost_table["in"]          / 1_000_000 +
        out_t * cost_table["out"]         / 1_000_000 +
        cw    * cost_table["cache_write"] / 1_000_000 +
        cr    * cost_table["cache_read"]  / 1_000_000
    )
    return {
        "input_tokens": in_t, "output_tokens": out_t,
        "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr,
        "cost_usd": round(dollars, 5), "model": model,
    }


def merge_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    """Sum a sequence of usage dicts (e.g. generate + verify1 + verify2)."""
    out = {"input_tokens": 0, "output_tokens": 0,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
           "cost_usd": 0.0, "model": "+".join(sorted({u.get("model","?") for u in usages if u}))}
    for u in usages:
        if not u: continue
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            out[k] += u.get(k, 0) or 0
        out["cost_usd"] += u.get("cost_usd", 0) or 0
    out["cost_usd"] = round(out["cost_usd"], 5)
    return out
