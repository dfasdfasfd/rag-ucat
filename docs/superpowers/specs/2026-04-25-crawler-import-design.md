# Crawler → Trainer Direct Import (No-OCR)

**Date:** 2026-04-25
**Status:** Design — awaiting approval
**Authors:** Ian (product), Claude (design)

## Goal

The Medify crawler (`/Users/ibutlerking/Documents/NGMP apps/crawler/`) becomes the preferred input source for the UCAT Trainer · RAG Edition (`/Users/ibutlerking/Documents/question generation app/`). Crawler output is written **directly in the trainer's JSON import schema** so the trainer's existing `⊕ Import JSON` button consumes it with no schema conversion and no vision-OCR pass.

## Why this shape

- **Cheaper than OCR.** The trainer currently has a `qwen2.5vl` vision-OCR ingester for screenshots. Skipping it removes the slowest and most error-prone step from the ingest path. The crawler already has clean DOM-extracted text (`stem`, `options[]`); we just need to capture the answer too.
- **Zero trainer schema changes.** The trainer's `db.import_json(...)` already handles the target shape. Reusing it avoids divergence between OCR-imported and crawler-imported docs in the KB.
- **Replayable.** A single `output/trainer-import.json` is the canonical bridge artefact. Crawler runs are reproducible; trainer imports are deterministic.

## Out of scope

- **Abstract Reasoning (AR)** — trainer has no AR schema. Skip with a warning. Adding AR support is a separate spec.
- **DM sub-type classification.** Trainer's DM schema requires `type ∈ {syllogism, logical, venn, probability, argument}`. We default everything to `"logical"` (valid catch-all). A real classifier is a future task.
- **Trainer GUI changes.** The existing `⊕ Import JSON` button is the entry point. We do not add a "Import from Crawler" button — the user just picks `output/trainer-import.json` from the crawler directory.

## Trainer schema recap (target shape)

The crawler must produce an array of these per-section docs (`src/config.py:41`, `src/quality.py:48`):

```jsonc
// VR
{ "section":"VR", "passage":"…(≥100 chars)…",
  "questions": [
    { "number":1, "text":"…", "type":"tf"|"mc",
      "options":{"A":"…","B":"…","C":"…","D":"…"},   // tf has only A/B/C
      "answer":"B", "explanation":"…" },
    …×4
  ]
}

// DM (standalone, 5 per doc)
{ "section":"DM",
  "questions": [
    { "number":1, "type":"logical", "text":"…",
      "options":{"A":"…","B":"…","C":"…","D":"…","E":"…"},
      "answer":"C", "explanation":"…" },
    …×5
  ]
}

// QR
{ "section":"QR", "stimulus":"…(≥20 chars, may be markdown table)…",
  "questions": [
    { "number":1, "text":"…",
      "options":{"A":"…","B":"…","C":"…","D":"…","E":"…"},
      "answer":"D", "explanation":"…" },
    …×4
  ]
}

// SJT
{ "section":"SJT", "scenario":"…(≥50 chars)…",
  "questions": [
    { "number":1, "text":"…",
      "options":{"A":"…","B":"…","C":"…","D":"…"},
      "answer":"A", "explanation":"…" },
    …×4
  ]
}
```

Validation rules (`src/quality.py:48–123`) the crawler output must satisfy:
- Required keys per section (passage / stimulus / scenario / questions).
- Exact question count (VR/QR/SJT = 4, DM = 5).
- No empty options, no duplicate option text, `answer` letter must exist in `options`.
- DM `type` must be in `valid_types`.

## Crawler changes

All changes live in `/Users/ibutlerking/Documents/NGMP apps/crawler/`.

### 1. Capture the Explain Answer panel

`src/capturer.ts` — after option capture, add a new step:

1. Click the "Explain Answer" control (top-left of each Medify question page).
2. Wait for the answer panel to render.
3. Read the revealed correct-answer letter and the explanation text.
4. Close the panel before navigating Next.

New selectors in `src/config/ucat-official.ts`:

```ts
explainAnswerButton:    "<TBD: e.g. button[aria-label='Explain Answer']>",
explainAnswerPanel:     "<TBD: panel container>",
explainAnswerLetter:    "<TBD: letter element inside panel>",
explainAnswerText:      "<TBD: explanation text inside panel>",
explainAnswerClose:     "<TBD: close button>",
```

Selector values are filled in during a one-time discovery pass against the live Medify UI (use Playwright codegen or DevTools). If the selectors miss at runtime, mark the entry `status: "partial"`, omit `correctAnswer`/`explanation`, and continue — don't crash the run.

### 2. Persist structured fields in `q####.json`

Today `q####.json` carries metadata only. Extend it with the structured content already extracted in `capturer.ts`:

```ts
// added fields on q####.json
stem: string;                            // passage + question text, DOM-extracted
options: Record<string, string>;         // { A: "…", B: "…", … }
correctAnswer?: string;                  // single letter, from Explain Answer panel
explanation?: string;                    // from Explain Answer panel
```

The existing concatenated `q####.txt` stays for archival/debugging.

### 3. Build trainer-schema bundles

New module: `src/trainer-bundle.ts`. After all captures complete in `runner.ts`, walk per-bucket entries and emit:

- `output/<bucket>/trainer.json` — array of trainer-schema docs for that bucket only (debugging aid).
- `output/trainer-import.json` — concatenation of all per-bucket arrays. **This is the file the trainer imports.**

Bundling rules per section:

| Section | Container field | Group size | Grouping rule                                      |
|---------|-----------------|-----------:|----------------------------------------------------|
| VR      | `passage`       | 4          | Common stem prefix across 4 consecutive captures   |
| QR      | `stimulus`      | 4          | Common stem prefix across 4 consecutive captures   |
| SJT     | `scenario`      | 4          | Common stem prefix across 4 consecutive captures   |
| DM      | (none)          | 5          | Sequential chunks of 5; each `stem` is a question  |
| AR      | —               | —          | Skip; emit `output/<bucket>/trainer.json.skipped`  |

### 4. Passage-extraction algorithm (VR/QR/SJT)

Each Medify capture's `stem` is `<passage> <question text>` concatenated. For four consecutive captures sharing the same passage:

1. Compute the longest common prefix (LCP) across the four stems by character.
2. Walk the LCP back to the nearest sentence boundary (`. `, `.\n`, `? `, `! `) so we don't slice mid-sentence. If no boundary in the last ~80 chars, fall back to the raw LCP.
3. `passage` = LCP (trimmed). Reject the group if `passage.length < min_passage_len` for VR (100) / `min_stimulus_len` for QR (20) / `min_scenario_len` for SJT (50).
4. Each question's `text` = its stem with the LCP stripped from the front, trimmed.
5. Tail group with fewer than 4 captures: drop and log.

This is the load-bearing piece. It needs unit tests with fixtures from real Medify text. Add tests in `tests/trainer-bundle.test.ts`.

### 5. Per-question field mapping

| Trainer field         | Source                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `number`              | 1-based position within the doc's `questions` array                    |
| `text`                | Stem with passage/stimulus/scenario LCP stripped (DM: full stem)       |
| `type` (VR)           | `"tf"` if `options` values match `{True, False, Can't Tell}` ci-insensitively, else `"mc"` |
| `type` (DM)           | `"logical"` (default — see Out of scope)                               |
| `options`             | `q####.json.options` map                                               |
| `answer`              | `q####.json.correctAnswer`. If missing, omit (validator allows missing)|
| `explanation`         | `q####.json.explanation`. If missing, omit                             |

For VR `tf` questions, validate `options` reduces to A/B/C only — drop a `D` if present (Medify's tf renders 3 radios, but a stale `D` would fail trainer validation).

### 6. Failure handling

- Per-question selector miss (Explain Answer): write `status: "partial"`, omit answer/explanation, continue.
- Group rejection (LCP too short, group incomplete): drop the doc, log to `output/trainer-bundle.log`.
- Crawler run remains successful (exit 0) as long as ≥1 valid trainer doc was emitted. If zero docs emit, exit non-zero with a clear error.

## Trainer changes

**None required for the happy path.** The user picks `output/trainer-import.json` via the existing `⊕ Import JSON` button.

Optional follow-up (not in this spec): record `source='crawler'` in `db.add_doc(...)` when the import path basename is `trainer-import.json`, so the KB tab can filter crawler-sourced docs. Cheap to add later; skip for now.

## Testing

- **Unit tests** in the crawler (`tests/`): LCP algorithm, sentence-boundary walk-back, VR `tf`/`mc` detection, DM 5-chunking, AR skip.
- **Fixture-based integration test**: drop a recorded set of `q####.json` files for one VR bucket into a fixture dir, run `trainer-bundle.ts`, assert the resulting `trainer-import.json` validates against the trainer schema (port `validate_schema` from Python or write a TS mirror — port is cleaner since it's the ground truth).
- **End-to-end smoke test** (manual): run `npm run crawl:dry` against a small Medify VR pack, open the trainer GUI, click `⊕ Import JSON`, pick `output/trainer-import.json`, click `⊛ Index Knowledge Base`, generate one VR question, confirm output looks UCAT-styled.

## Risks & mitigations

| Risk                                                                                  | Mitigation                                                                 |
|---------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Explain Answer selectors break when Medify ships UI changes                           | Crawler `status: "partial"` path keeps capture working without answers; selector discovery is a quick maintenance task |
| LCP algorithm misgroups questions (e.g. two passages share a generic intro)            | Reject groups where passage < min length; log dropped groups for review    |
| DM `type="logical"` default reduces retrieval diversity for DM generations            | Acceptable for v1 (DM diversity hint is `data_type`, which RAG indexes from explanation/stem); revisit when adding a classifier |
| Crawler captures duplicates across re-runs                                            | The crawler's per-question `id` is deterministic (`bucket+index`); the bundle emitter dedups by `id` before writing. Trainer-side dedup against existing KB docs runs at *generation* time (`QUALITY_THRESHOLDS.dedup_kb = 0.92`), not import time, so we do not rely on it. |

## File touch list (estimate)

**Crawler (new + modified):**
- `src/capturer.ts` — modify (add Explain Answer capture + structured JSON fields)
- `src/config/ucat-official.ts` — modify (5 new selectors)
- `src/types.ts` — modify (extend `CapturedQuestionExtras` with `stem`, `options`, `correctAnswer`, `explanation`)
- `src/trainer-bundle.ts` — new (grouping + LCP + bundle emit)
- `src/runner.ts` — modify (call `trainer-bundle` after all captures)
- `tests/trainer-bundle.test.ts` — new
- `tests/lcp.test.ts` — new

**Trainer:**
- No changes in this spec.
