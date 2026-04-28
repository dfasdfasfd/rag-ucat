# UCAT Trainer  ·  RAG Edition

Generates unlimited UCAT questions **grounded only in your knowledge base**.
Everything runs locally — no internet required after setup.

---

## How RAG works here

```
Your Questions (JSON)
        │
        ▼
  [Embedding Model]  ← nomic-embed-text (Ollama)
        │  converts each doc to a semantic vector
        ▼
  [Vector Store]     ← SQLite on your machine
        │
        │   at generation time:
        │   1. embed section + topic hint → query vector
        │   2. cosine similarity → find top-4 most relevant docs
        │   3. pass only those docs as context to the LLM
        ▼
  [LLM]              ← llama3.2 / mistral (Ollama)
        │  sees ONLY your retrieved docs — nothing else shapes the format
        ▼
  New UCAT Question Set
```

---

## Setup (one-time, ~10 minutes)

1. Install Python 3.8+: https://python.org
2. Install Ollama: https://ollama.ai
3. Open a terminal and run:
   ollama pull llama3.2
   ollama pull nomic-embed-text
4. Run the app:
   python ucat_trainer.py

No pip installs. No API keys. Works offline.

---

## Workflow

1. Add Samples or Import JSON → adds docs to knowledge base
2. Index Knowledge Base → embeds all docs into vectors (do this after every import)
3. Generate → pick section, optionally type a topic hint, click Generate
4. Review output → Save good ones back to KB → Re-index → Better future generations

---

## Topic hints

Type "ecology", "probability", "medical ethics" etc. to steer which documents are retrieved.
The retrieval query becomes: "UCAT Verbal Reasoning question about ecology"

---

## JSON import format

[
  {
    "section": "VR",
    "passage": "...",
    "questions": [
      {
        "number": 1, "text": "...", "type": "tf",
        "minigame_kind": "tfc",
        "options": {"A": "True", "B": "False", "C": "Can't Tell"},
        "answer": "B", "explanation": "..."
      }
    ]
  }
]

Section codes: VR  DM  QR  SJT

## minigame_kind

Each question carries a `minigame_kind` tag that routes it into a specific
Pocket UCAT minigame on the consumer side. Allowed values per section:

- VR  : tfc, main-idea, paraphrase, tone-purpose, inference
- DM  : syllogism, logic-grid, venn, probability, argument-strength, assumption
- QR  : rapid-estimation, data-table, ratio, fraction, graph-grab, chart-sprint
- SJT : appropriateness, importance, empathy, escalation, role-identification

If `minigame_kind` is missing the Pocket UCAT importer falls back to keyword
heuristics, so old generations remain valid — but populating it is preferred
because the heuristic is ~70% accurate.

---

## Troubleshooting

Ollama offline → run: ollama serve
No models → ollama pull llama3.2 && ollama pull nomic-embed-text
No indexed docs → add docs first, then click Index Knowledge Base
JSON errors → try a larger model (mistral) or generate again
Slow → normal on CPU (30-90s). GPU makes it 10x faster.
