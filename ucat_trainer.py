#!/usr/bin/env python3
"""
UCAT Trainer  ·  RAG Edition
The model generates questions grounded ONLY in your knowledge base.
Requires: Python 3.8+  ·  Ollama (https://ollama.ai)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import sqlite3
import json
import math
import re
import requests
import threading
import textwrap
from datetime import datetime

# ─── Constants ────────────────────────────────────────────────────────────────

APP_TITLE     = "UCAT Trainer  ·  RAG"
DB_FILE       = "ucat_rag.db"
OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_LLM   = "llama3.2"
DEFAULT_EMBED = "nomic-embed-text"
TOP_K         = 4

SECTIONS = {
    "VR": "Verbal Reasoning",
    "DM": "Decision Making",
    "QR": "Quantitative Reasoning",
    "AR": "Abstract Reasoning",
}
SECTION_COLORS = {
    "VR": "#4A90D9", "DM": "#E8943A",
    "QR": "#3FB950", "AR": "#A78BFA",
}

SECTION_SCHEMAS = {
    "VR": {
        "desc": "A passage (200-300 words) followed by exactly 4 questions. Each question is either True/False/Can't Tell OR 4-option multiple choice (A-D). Questions must be answerable ONLY from the passage.",
        "shape": ('{"section":"VR","passage":"...","questions":['
                  '{"number":1,"text":"...","type":"tf","options":{"A":"True","B":"False","C":"Can\'t Tell"},"answer":"A","explanation":"..."},'
                  '{"number":2,"text":"...","type":"mc","options":{"A":"...","B":"...","C":"...","D":"..."},"answer":"C","explanation":"..."}]}'),
    },
    "DM": {
        "desc": "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), venn (set relationships), probability, or argument (strongest argument for/against). Each has 5 options (A-E).",
        "shape": ('{"section":"DM","questions":['
                  '{"number":1,"type":"syllogism","text":"...","options":{"A":"...","B":"...","C":"...","D":"...","E":"..."},"answer":"B","explanation":"..."}]}'),
    },
    "QR": {
        "desc": "One data stimulus (table or chart as markdown text) followed by exactly 4 calculation questions. Each question has 5 numerical options (A-E). Include step-by-step working in each explanation.",
        "shape": ('{"section":"QR","stimulus":"Title\\n\\n| Col | Col |\\n|-----|-----|\\n| val | val |",'
                  '"questions":[{"number":1,"text":"...","options":{"A":"12","B":"14","C":"16","D":"18","E":"20"},"answer":"C","explanation":"step...=16"}]}'),
    },
    "AR": {
        "desc": "Type 1 set. Describe Set A (6 panels, hidden rule using Unicode shapes: ■ □ ▲ △ ○ ●). Describe Set B (6 panels, different hidden rule). Then 5 test shapes answered Set A / Set B / Neither.",
        "shape": ('{"section":"AR","set_a_description":"Panel 1: ...\\nHidden rule: ...","set_b_description":"Panel 1: ...\\nHidden rule: ...",'
                  '"questions":[{"number":1,"text":"Test shape: ...","options":{"A":"Set A","B":"Set B","C":"Neither"},"answer":"A","explanation":"..."}]}'),
    },
}

# ─── Utilities ────────────────────────────────────────────────────────────────

def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma  = math.sqrt(sum(x * x for x in a))
    mb  = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


def embed_text_for(data: dict) -> str:
    parts = [f"UCAT {SECTIONS.get(data.get('section',''), '')} question"]
    for k in ("passage", "stimulus", "set_a_description", "set_b_description"):
        if k in data:
            parts.append(str(data[k])[:500])
    for q in data.get("questions", [])[:3]:
        parts.append(q.get("text", ""))
    return " ".join(parts)


def format_qset(data: dict) -> str:
    sec   = data.get("section", "?")
    lines = ["═" * 64, f"  {SECTIONS.get(sec, sec)}  ·  Question Set", "═" * 64, ""]
    for key, label in (("passage", "PASSAGE"), ("stimulus", "DATA / STIMULUS")):
        if key in data:
            lines += [label, "─" * 40]
            for para in str(data[key]).split("\n"):
                lines.append(textwrap.fill(para, 72) if para.strip() else "")
            lines.append("")
    for key, label in (("set_a_description", "SET A"), ("set_b_description", "SET B")):
        if key in data:
            lines += [label, "─" * 40, str(data[key]), ""]
    for q in data.get("questions", []):
        lines.append(f"Q{q.get('number','?')}.  {q.get('text','')}")
        for k, v in q.get("options", {}).items():
            lines.append(f"       {k})  {v}")
        lines.append(f"       ✓  Answer: {q.get('answer','?')}")
        if q.get("explanation"):
            lines.append(textwrap.fill(
                q["explanation"], 64,
                initial_indent="       💡  ", subsequent_indent="           "
            ))
        lines.append("")
    return "\n".join(lines)


def parse_json(text: str):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ─── Database ────────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path: str = DB_FILE):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS kb (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            section     TEXT NOT NULL,
            source      TEXT DEFAULT 'manual',
            data        TEXT NOT NULL,
            embed_text  TEXT NOT NULL,
            embedding   TEXT DEFAULT NULL,
            embed_model TEXT DEFAULT NULL,
            created     TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS generated (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            section     TEXT NOT NULL,
            data        TEXT NOT NULL,
            context_ids TEXT NOT NULL,
            created     TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()

    # ── KB ──

    def add_doc(self, section: str, data: dict, source: str = "manual") -> int:
        et = embed_text_for(data)
        c  = self.conn.cursor()
        c.execute("INSERT INTO kb (section,source,data,embed_text) VALUES (?,?,?,?)",
                  (section, source, json.dumps(data), et))
        self.conn.commit()
        return c.lastrowid

    def set_embedding(self, doc_id: int, vec: list, model: str):
        c = self.conn.cursor()
        c.execute("UPDATE kb SET embedding=?,embed_model=? WHERE id=?",
                  (json.dumps(vec), model, doc_id))
        self.conn.commit()

    def get_all_docs(self, section=None, limit=5000):
        c = self.conn.cursor()
        q = ("SELECT id,section,source,data,embed_text,embedding,embed_model,created FROM kb"
             + (f" WHERE section=?" if section else "")
             + " ORDER BY created DESC LIMIT ?")
        args = (section, limit) if section else (limit,)
        return [
            {"id": r[0], "section": r[1], "source": r[2],
             "data": json.loads(r[3]), "embed_text": r[4],
             "embedding": json.loads(r[5]) if r[5] else None,
             "embed_model": r[6], "created": r[7]}
            for r in c.execute(q, args).fetchall()
        ]

    def get_unindexed(self, section=None):
        c = self.conn.cursor()
        if section:
            rows = c.execute("SELECT id,embed_text FROM kb WHERE embedding IS NULL AND section=?", (section,)).fetchall()
        else:
            rows = c.execute("SELECT id,embed_text FROM kb WHERE embedding IS NULL").fetchall()
        return [{"id": r[0], "embed_text": r[1]} for r in rows]

    def count(self, section=None, indexed_only=False):
        c    = self.conn.cursor()
        cond = ("WHERE section=?" if section else "") + (" AND embedding IS NOT NULL" if indexed_only and section else " WHERE embedding IS NOT NULL" if indexed_only else "")
        args = (section,) if section else ()
        return c.execute(f"SELECT COUNT(*) FROM kb {cond}", args).fetchone()[0]

    def retrieve(self, section: str, qvec: list, top_k: int = TOP_K):
        docs = [d for d in self.get_all_docs(section) if d["embedding"]]
        if not docs:
            return []
        scored = sorted(
            [(cosine_sim(qvec, d["embedding"]), d) for d in docs],
            key=lambda x: x[0], reverse=True
        )
        return scored[:top_k]

    def import_json(self, path: str) -> int:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else [raw]
        n = 0
        for item in items:
            if isinstance(item, dict) and item.get("section") in SECTIONS:
                self.add_doc(item["section"], item, source="imported")
                n += 1
        return n

    # ── Generated ──

    def add_generated(self, section: str, data: dict, ctx_ids: list) -> int:
        c = self.conn.cursor()
        c.execute("INSERT INTO generated (section,data,context_ids) VALUES (?,?,?)",
                  (section, json.dumps(data), json.dumps(ctx_ids)))
        self.conn.commit()
        return c.lastrowid

    def get_generated(self, limit: int = 500):
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id,section,data,context_ids,created FROM generated ORDER BY created DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [{"id": r[0], "section": r[1], "data": json.loads(r[2]),
                 "context_ids": json.loads(r[3]), "created": r[4]} for r in rows]

    def promote_to_kb(self, gen_id: int):
        c = self.conn.cursor()
        row = c.execute("SELECT section,data FROM generated WHERE id=?", (gen_id,)).fetchone()
        if row:
            self.add_doc(row[0], json.loads(row[1]), source="generated")
            return True
        return False

    def close(self):
        self.conn.close()


# ─── Ollama ───────────────────────────────────────────────────────────────────

class Ollama:
    base = OLLAMA_BASE

    @classmethod
    def is_up(cls) -> bool:
        try:
            return requests.get(f"{cls.base}/api/tags", timeout=3).ok
        except Exception:
            return False

    @classmethod
    def list_models(cls):
        try:
            r = requests.get(f"{cls.base}/api/tags", timeout=4)
            if r.ok:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    @classmethod
    def embed(cls, text: str, model: str) -> list:
        r = requests.post(f"{cls.base}/api/embeddings",
                          json={"model": model, "prompt": text}, timeout=60)
        r.raise_for_status()
        return r.json()["embedding"]

    @classmethod
    def generate(cls, system: str, prompt: str, model: str,
                 temperature: float = 0.82, num_predict: int = 2800) -> str:
        r = requests.post(
            f"{cls.base}/api/generate",
            json={"model": model, "system": system, "prompt": prompt,
                  "stream": False, "options": {"temperature": temperature,
                                               "top_p": 0.92, "num_predict": num_predict}},
            timeout=240
        )
        r.raise_for_status()
        return r.json().get("response", "")


# ─── RAG Engine ───────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self, db: Database, llm: str = DEFAULT_LLM, emb: str = DEFAULT_EMBED):
        self.db  = db
        self.llm = llm
        self.emb = emb

    def index_doc(self, doc_id: int, text: str):
        vec = Ollama.embed(text, self.emb)
        self.db.set_embedding(doc_id, vec, self.emb)

    def index_all(self, section=None, on_progress=None):
        pending = self.db.get_unindexed(section)
        for i, doc in enumerate(pending):
            if on_progress:
                on_progress(i, len(pending))
            self.index_doc(doc["id"], doc["embed_text"])
        return len(pending)

    def retrieve(self, section: str, hint: str = ""):
        query = f"UCAT {SECTIONS[section]} question"
        if hint.strip():
            query += f" about {hint.strip()}"
        try:
            qvec = Ollama.embed(query, self.emb)
        except Exception:
            import random
            docs = self.db.get_all_docs(section)
            random.shuffle(docs)
            return [(0.0, d) for d in docs[:TOP_K]]
        return self.db.retrieve(section, qvec, TOP_K)

    def _system(self, section: str, retrieved: list) -> str:
        s = SECTION_SCHEMAS[section]
        ex = ""
        if retrieved:
            blocks = [
                f"--- KNOWLEDGE BASE DOC {i+1} (similarity {sc:.3f}) ---\n{json.dumps(d['data'], indent=2)}"
                for i, (sc, d) in enumerate(retrieved)
            ]
            ex = (
                "\n\nThe following documents are from the user's knowledge base. "
                "They define the authoritative format, difficulty, and style you MUST follow. "
                "Your output must mirror their JSON structure exactly. "
                "Do NOT use any format knowledge outside these documents.\n\n"
                + "\n\n".join(blocks)
            )
        return (
            f"You are a UCAT {SECTIONS[section]} question generator.\n"
            f"Generate: {s['desc']}\n\n"
            f"RETURN ONLY a single valid JSON object — no preamble, no markdown fences.\n"
            f"JSON schema to follow exactly:\n{s['shape']}"
            + ex
        )

    def generate(self, section: str, hint: str = "", on_progress=None):
        if on_progress: on_progress("Embedding retrieval query…")
        retrieved = self.retrieve(section, hint)

        if on_progress: on_progress(f"Retrieved {len(retrieved)} doc(s). Building prompt…")
        system = self._system(section, retrieved)

        user = (
            f"Generate a completely NEW and UNIQUE UCAT {SECTIONS[section]} question set. "
            + (f"Focus on the topic: {hint.strip()}. " if hint.strip() else "")
            + "Content, numbers, scenarios, and passages must be entirely original. "
            "Follow the format of the reference documents precisely. "
            "Return ONLY valid JSON."
        )

        if on_progress: on_progress("Generating with local AI model…")
        raw  = Ollama.generate(system, user, self.llm)

        if on_progress: on_progress("Parsing output…")
        data = parse_json(raw)
        if data is None:
            raise RuntimeError(
                "Model returned output that couldn't be parsed as JSON.\n\n"
                "Suggestions:\n"
                "• Try a larger model (mistral, llama3.1)\n"
                "• Add more documents to the knowledge base\n"
                "• Try generating again (random temperature variation often fixes this)"
            )
        data["section"] = section
        ctx_ids = [d["id"] for _, d in retrieved]
        self.db.add_generated(section, data, ctx_ids)
        return data, retrieved


# ─── Built-in sample documents ───────────────────────────────────────────────

SAMPLES = [
    {"section": "VR",
     "passage": (
         "The urban heat island (UHI) effect describes the phenomenon whereby metropolitan areas experience "
         "markedly higher temperatures than their surrounding rural regions. This disparity arises primarily "
         "from the replacement of natural vegetation and permeable surfaces with concrete, asphalt, and "
         "buildings — materials that absorb solar radiation during the day and re-radiate it as heat overnight, "
         "preventing the natural cooling that occurs in vegetated landscapes. Waste heat from vehicles, "
         "industrial processes, and air conditioning systems compounds the effect. Consequences include "
         "elevated energy demand for cooling, deterioration of air quality through increased ground-level "
         "ozone formation, greater frequency of heat-related illness, and impaired stormwater management. "
         "Mitigation strategies under investigation include the planting of urban trees and green roofs, "
         "the adoption of high-albedo 'cool' paving materials that reflect rather than absorb sunlight, "
         "and redesigning street grids to promote ventilating airflows. Research indicates that doubling "
         "urban tree canopy cover could reduce peak summer temperatures by between 2°C and 8°C depending "
         "on local conditions, representing one of the most cost-effective adaptation measures available."
     ),
     "questions": [
         {"number": 1, "text": "Air conditioning systems contribute to the UHI effect.", "type": "tf",
          "options": {"A": "True", "B": "False", "C": "Can't Tell"}, "answer": "A",
          "explanation": "The passage explicitly lists waste heat from air conditioning as a contributing factor."},
         {"number": 2, "text": "Urban areas are only warmer than rural areas during the daytime.", "type": "tf",
          "options": {"A": "True", "B": "False", "C": "Can't Tell"}, "answer": "B",
          "explanation": "Materials re-radiate heat overnight, so the UHI effect persists at night."},
         {"number": 3, "text": "Which best describes the primary cause of the UHI effect?", "type": "mc",
          "options": {"A": "Industrial air pollution trapping heat",
                      "B": "Replacement of natural surfaces with heat-absorbing materials",
                      "C": "Increased vehicle exhaust in city centres",
                      "D": "Deforestation in rural areas surrounding cities"},
          "answer": "B",
          "explanation": "The passage identifies this replacement as the primary cause in its opening sentences."},
         {"number": 4, "text": "Doubling tree canopy cover guarantees an 8°C temperature reduction.", "type": "tf",
          "options": {"A": "True", "B": "False", "C": "Can't Tell"}, "answer": "B",
          "explanation": "The passage says 'between 2°C and 8°C depending on local conditions' — 8°C is a maximum, not guaranteed."},
     ]},
    {"section": "VR",
     "passage": (
         "Cognitive load theory, developed by John Sweller in the 1980s, holds that effective instruction "
         "must account for the finite capacity of human working memory. The theory distinguishes three "
         "categories. Intrinsic load is determined by the inherent complexity of the material and cannot "
         "be reduced without changing the content. Extraneous load arises from poor instructional design — "
         "for example, requiring learners to mentally integrate information needlessly distributed across "
         "a diagram and its accompanying text. Germane load represents the cognitive effort invested in "
         "constructing and storing schemas: organised mental frameworks enabling faster future processing. "
         "Effective teaching minimises extraneous load so that resources can be directed toward schema "
         "formation. Key applications include the use of worked examples for novice learners — reducing "
         "the effort of problem-solving search — and the expertise reversal effect, which observes that "
         "formats beneficial to beginners may impede expert learners by imposing redundant processing on "
         "automated knowledge. Critics have questioned whether the three load types can be cleanly measured "
         "and whether working memory capacity is truly as fixed as originally proposed."
     ),
     "questions": [
         {"number": 1, "text": "Germane load refers to the cognitive effort caused by poor instructional design.", "type": "tf",
          "options": {"A": "True", "B": "False", "C": "Can't Tell"}, "answer": "B",
          "explanation": "Poor design causes extraneous load. Germane load is the effort invested in schema formation."},
         {"number": 2, "text": "Cognitive load theory was proposed before the year 1990.", "type": "tf",
          "options": {"A": "True", "B": "False", "C": "Can't Tell"}, "answer": "A",
          "explanation": "The passage states it was developed 'in the 1980s', which is before 1990."},
         {"number": 3, "text": "What does the expertise reversal effect describe?", "type": "mc",
          "options": {"A": "Worked examples benefit all learners equally",
                      "B": "Instructional formats helpful to novices may hinder expert learners",
                      "C": "Experts should always be taught with complex problems",
                      "D": "Novices learn faster than experts in all contexts"},
          "answer": "B",
          "explanation": "The passage defines expertise reversal exactly as option B describes."},
         {"number": 4, "text": "All researchers agree that working memory capacity is fixed.", "type": "tf",
          "options": {"A": "True", "B": "False", "C": "Can't Tell"}, "answer": "B",
          "explanation": "The final sentence notes critics have questioned whether capacity is 'truly as fixed as originally proposed'."},
     ]},
    {"section": "DM",
     "questions": [
         {"number": 1, "type": "syllogism",
          "text": "Premise 1: All surgeons are doctors.\nPremise 2: Some doctors work night shifts.\n\nWhich conclusion follows?",
          "options": {"A": "All surgeons work night shifts", "B": "Some surgeons work night shifts",
                      "C": "No surgeons work night shifts", "D": "Some doctors are not surgeons",
                      "E": "None of the above"},
          "answer": "E",
          "explanation": "The 'some doctors' who work nights may or may not overlap with surgeons. No conclusion about surgeons can be drawn."},
         {"number": 2, "type": "logical",
          "text": ("Five friends — Alice, Ben, Cara, Dan, Eve — each own one pet: cat, dog, rabbit, hamster, fish.\n"
                   "• Alice does not own a cat or dog.\n• Ben owns the rabbit.\n"
                   "• Cara owns the fish.\n• Dan does not own the hamster.\n"
                   "Which must be true?"),
          "options": {"A": "Alice owns the hamster", "B": "Dan owns the cat",
                      "C": "Eve owns the dog", "D": "Alice owns the fish", "E": "Eve owns the hamster"},
          "answer": "A",
          "explanation": "Ben=rabbit, Cara=fish. Alice cannot have cat/dog/rabbit/fish, so Alice=hamster. Dan cannot have hamster/rabbit/fish, so Dan has cat or dog. Eve gets the remaining."},
         {"number": 3, "type": "probability",
          "text": "A bag contains 6 red, 4 blue, and 2 yellow counters. One drawn at random. P(NOT red)?",
          "options": {"A": "1/6", "B": "1/3", "C": "1/2", "D": "2/3", "E": "5/6"},
          "answer": "C",
          "explanation": "Non-red = 4+2 = 6. Total = 12. P = 6/12 = 1/2."},
         {"number": 4, "type": "argument",
          "text": "Statement: The voting age should be lowered to 16.\nStrongest argument IN FAVOUR?",
          "options": {
              "A": "Teenagers are affected by government policy and pay tax, yet have no democratic voice",
              "B": "Some 16-year-olds are more mature than some adults",
              "C": "Other countries have adopted this policy",
              "D": "It would increase overall voter turnout",
              "E": "Young people are more engaged with social media"},
          "answer": "A",
          "explanation": "A directly links democratic rights to civic obligations already borne by 16-year-olds — the most principled argument."},
         {"number": 5, "type": "venn",
          "text": ("All chefs are creative. Some creative people are musicians. No musicians are accountants.\n"
                   "Which must be true?"),
          "options": {"A": "Some chefs are musicians", "B": "No chefs are accountants",
                      "C": "No accountants are creative", "D": "Some creative people are not accountants",
                      "E": "All musicians are chefs"},
          "answer": "D",
          "explanation": "Creative musicians exist (given), and no musicians are accountants, so those creative-musicians are not accountants. Hence some creative people are not accountants."},
     ]},
    {"section": "QR",
     "stimulus": (
         "Monthly Sales — TechStore UK (£000s)\n\n"
         "| Category    | Jan | Feb | Mar | Apr |\n"
         "|-------------|-----|-----|-----|-----|\n"
         "| Laptops     |  84 |  91 | 107 |  98 |\n"
         "| Phones      |  62 |  58 |  74 |  89 |\n"
         "| Accessories |  23 |  27 |  31 |  28 |\n"
         "| Gaming      |  45 |  39 |  52 |  61 |\n"
         "| **Total**   | 214 | 215 | 264 | 276 |"
     ),
     "questions": [
         {"number": 1, "text": "What is the mean monthly total sales across the four months?",
          "options": {"A": "£232,250", "B": "£238,000", "C": "£242,250", "D": "£244,000", "E": "£248,500"},
          "answer": "C", "explanation": "(214+215+264+276)/4 = 969/4 = 242.25 → £242,250"},
         {"number": 2, "text": "Phones sales grew by what percentage from February to April?",
          "options": {"A": "43.5%", "B": "48.3%", "C": "51.2%", "D": "53.4%", "E": "56.7%"},
          "answer": "D", "explanation": "(89−58)/58 × 100 = 31/58 × 100 = 53.4%"},
         {"number": 3, "text": "In March, Accessories were what fraction of total sales?",
          "options": {"A": "31/264", "B": "1/9", "C": "1/8", "D": "7/56", "E": "1/7"},
          "answer": "A", "explanation": "31/264 — this does not simplify to any other option."},
         {"number": 4, "text": "If Gaming grows 20% from April, what will May Gaming sales be?",
          "options": {"A": "£68,200", "B": "£70,800", "C": "£71,400", "D": "£73,200", "E": "£75,600"},
          "answer": "D", "explanation": "61 × 1.20 = 73.2 → £73,200"},
     ]},
    {"section": "AR",
     "set_a_description": (
         "Panel 1: ■ ■ ○  (2 black squares, 1 white circle)\n"
         "Panel 2: ■ ○ ○  (1 black square, 2 white circles)\n"
         "Panel 3: ■ ■ ■ ○  (3 black squares, 1 white circle)\n"
         "Panel 4: ■ ○  (1 black square, 1 white circle)\n"
         "Panel 5: ■ ■ ○ ○  (2 black squares, 2 white circles)\n"
         "Panel 6: ■ ■ ■ ○ ○  (3 black squares, 2 white circles)\n"
         "Hidden rule: Each panel contains at least one ■ AND at least one ○. No other shapes."
     ),
     "set_b_description": (
         "Panel 1: ▲ ▲ ▲  (3 triangles)\n"
         "Panel 2: ▲ ▲  (2 triangles)\n"
         "Panel 3: ▲ ▲ ▲ ▲  (4 triangles)\n"
         "Panel 4: ▲  (1 triangle)\n"
         "Panel 5: ▲ ▲ ▲ ▲ ▲  (5 triangles)\n"
         "Panel 6: ▲ ▲ ▲ ▲ ▲ ▲  (6 triangles)\n"
         "Hidden rule: Each panel contains ONLY triangles — any number, no other shapes."
     ),
     "questions": [
         {"number": 1, "text": "Test shape: ■ ■ ■  (3 black squares, no circles)",
          "options": {"A": "Set A", "B": "Set B", "C": "Neither"}, "answer": "C",
          "explanation": "Has squares but no circle — fails Set A's requirement for at least one circle."},
         {"number": 2, "text": "Test shape: ▲ ▲  (2 black triangles only)",
          "options": {"A": "Set A", "B": "Set B", "C": "Neither"}, "answer": "B",
          "explanation": "Contains only triangles — matches Set B's rule."},
         {"number": 3, "text": "Test shape: ■ ○ ○ ○  (1 black square and 3 white circles)",
          "options": {"A": "Set A", "B": "Set B", "C": "Neither"}, "answer": "A",
          "explanation": "Contains at least one square and at least one circle — matches Set A."},
         {"number": 4, "text": "Test shape: ▲ ○  (1 triangle and 1 circle)",
          "options": {"A": "Set A", "B": "Set B", "C": "Neither"}, "answer": "C",
          "explanation": "No square (fails Set A). Not only triangles (circle present, fails Set B)."},
         {"number": 5, "text": "Test shape: □ ○  (1 white square and 1 white circle)",
          "options": {"A": "Set A", "B": "Set B", "C": "Neither"}, "answer": "C",
          "explanation": "Square is white not black — Set A requires black squares specifically."},
     ]},
]

# ─── Theme ────────────────────────────────────────────────────────────────────

BG = "#0C1117"; PANEL = "#161B22"; PANEL2 = "#1C2128"; BORDER = "#30363D"
ACCENT = "#58A6FF"; TEXT = "#E6EDF3"; MUTED = "#7D8590"
SUCCESS = "#3FB950"; DANGER = "#F85149"; WARN = "#D29922"

FM = ("Courier New", 11); FB = ("Courier New", 11, "bold")
FS = ("Courier New", 9);  FSB = ("Courier New", 9, "bold")
FT = ("Courier New", 15, "bold"); FH = ("Courier New", 10, "bold")


def mkbtn(parent, text, cmd, fg="white", bg=ACCENT, font=None, **kw):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                     font=font or FB, relief="flat", cursor="hand2",
                     activebackground=bg, activeforeground=fg, **kw)


# ─── Application ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1340x880")
        self.minsize(1020, 680)
        self.configure(bg=BG)

        self.db  = Database()
        self.rag = RAGEngine(self.db)

        self._last_data      = None
        self._last_section   = None
        self._last_retrieved = []
        self._sel_gen_id     = None

        self._style()
        self._ui()
        self.after(700, self._chk_ollama)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT, font=FM)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, font=FM, padding=[18, 9])
        s.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", ACCENT)])
        s.configure("Treeview", background=PANEL, foreground=TEXT,
                    fieldbackground=PANEL, font=FS, rowheight=28, borderwidth=0)
        s.configure("Treeview.Heading", background=BG, foreground=MUTED, font=FH)
        s.map("Treeview", background=[("selected", "#1F6FEB")], foreground=[("selected", "white")])
        s.configure("TScrollbar", background=PANEL, troughcolor=BG, borderwidth=0)
        s.configure("TCombobox", font=FM)
        s.configure("TSeparator", background=BORDER)

    # ── Skeleton ──────────────────────────────────────────────────────────────

    def _ui(self):
        # Header
        hdr = tk.Frame(self, bg=PANEL, height=54)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  UCAT TRAINER", bg=PANEL, fg=ACCENT,
                 font=("Courier New", 15, "bold")).pack(side="left", padx=22, pady=12)
        tk.Label(hdr, text="RAG", bg=PANEL, fg=WARN, font=FSB).pack(side="left", pady=12)

        self._dot  = tk.Label(hdr, text="●", bg=PANEL, fg=DANGER, font=("Courier New", 13))
        self._dlbl = tk.Label(hdr, text="Checking Ollama…", bg=PANEL, fg=MUTED, font=FS)
        self._dot.pack(side="right", padx=(0, 18))
        self._dlbl.pack(side="right", padx=(0, 4))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self._sidebar(body)

        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True)
        self._t_gen = tk.Frame(nb, bg=BG)
        self._t_kb  = tk.Frame(nb, bg=BG)
        self._t_out = tk.Frame(nb, bg=BG)
        nb.add(self._t_gen, text="  ⚡  GENERATE  ")
        nb.add(self._t_kb,  text="  🗄  KNOWLEDGE BASE  ")
        nb.add(self._t_out, text="  📋  OUTPUT HISTORY  ")
        self._tab_gen()
        self._tab_kb()
        self._tab_out()

        self._sbar = tk.Label(self, text="Ready", bg="#090C10", fg=MUTED, font=FS, anchor="w")
        self._sbar.pack(fill="x", padx=12, pady=(2, 4))
        self._refresh_stats()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=PANEL, width=220)
        sb.pack(side="left", fill="y"); sb.pack_propagate(False)

        tk.Label(sb, text="KNOWLEDGE BASE", bg=PANEL, fg=MUTED,
                 font=FSB).pack(anchor="w", padx=16, pady=(20, 6))

        self._slabels = {}
        for code in SECTIONS:
            row = tk.Frame(sb, bg=PANEL)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text="■", bg=PANEL, fg=SECTION_COLORS[code],
                     font=("Courier New", 12)).pack(side="left", padx=(4, 8))
            tk.Label(row, text=code, bg=PANEL, fg=TEXT, font=FB).pack(side="left")
            lbl = tk.Label(row, text="0/0", bg=PANEL, fg=MUTED, font=FS)
            lbl.pack(side="right", padx=8)
            self._slabels[code] = lbl
        tk.Label(sb, text="indexed / total", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="e", padx=14, pady=(0, 4))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=12)

        tk.Label(sb, text="LLM MODEL", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16, pady=(0, 4))
        self._llm_var = tk.StringVar(value=DEFAULT_LLM)
        self._llm_cb  = ttk.Combobox(sb, textvariable=self._llm_var, width=19, state="readonly")
        self._llm_cb.pack(padx=14, pady=(0, 8))

        tk.Label(sb, text="EMBED MODEL", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16, pady=(0, 4))
        self._emb_var = tk.StringVar(value=DEFAULT_EMBED)
        self._emb_cb  = ttk.Combobox(sb, textvariable=self._emb_var, width=19, state="readonly")
        self._emb_cb.pack(padx=14, pady=(0, 6))

        mkbtn(sb, "↻  Refresh Models", self._refresh_models,
              bg=PANEL, fg=MUTED, font=FS, pady=4
              ).pack(padx=14, pady=(0, 12), fill="x")

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)

        self._idx_btn  = mkbtn(sb, "⊛  Index Knowledge Base", self._do_index,
                                bg="#1F6FEB", pady=8)
        self._idx_btn.pack(padx=14, pady=(10, 2), fill="x")
        self._idx_lbl = tk.Label(sb, text="", bg=PANEL, fg=MUTED, font=FS)
        self._idx_lbl.pack(padx=14, anchor="w", pady=(0, 10))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)

        mkbtn(sb, "⊕  Import JSON",   self._import,      bg="#2D7D46", pady=7
              ).pack(padx=14, pady=3, fill="x")
        mkbtn(sb, "⊞  Add Samples",   self._add_samples, bg=PANEL, fg=ACCENT,
              font=FS, pady=6).pack(padx=14, pady=2, fill="x")
        mkbtn(sb, "↓  Export Output", self._export,      bg=PANEL, fg=MUTED,
              font=FS, pady=6).pack(padx=14, pady=2, fill="x")

    # ── Generate tab ──────────────────────────────────────────────────────────

    def _tab_gen(self):
        p = tk.Frame(self._t_gen, bg=BG)
        p.pack(fill="both", expand=True, padx=36, pady=24)

        tk.Label(p, text="Generate New Questions", bg=BG, fg=TEXT, font=FT).pack(anchor="w")
        tk.Label(p,
                 text="Retrieves the most relevant docs from your knowledge base and generates grounded questions.",
                 bg=BG, fg=MUTED, font=FS).pack(anchor="w", pady=(2, 16))

        # Section
        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 12))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._gsec = tk.StringVar(value="VR")
        for code in SECTIONS:
            tk.Radiobutton(
                sr, text=f" {code} ", variable=self._gsec, value=code,
                bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                activeforeground=ACCENT, font=FB, indicatoron=False,
                relief="flat", bd=1, padx=12, pady=6, cursor="hand2"
            ).pack(side="left", padx=4)

        # Hint
        hr = tk.Frame(p, bg=BG); hr.pack(fill="x", pady=(0, 12))
        tk.Label(hr, text="Topic hint:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._hint = tk.StringVar()
        tk.Entry(hr, textvariable=self._hint, bg=PANEL2, fg=TEXT, font=FM,
                 insertbackground=ACCENT, relief="flat", width=34).pack(side="left")
        tk.Label(hr, text="(optional — steers retrieval, e.g. 'ecology', 'probability')",
                 bg=BG, fg=MUTED, font=FS).pack(side="left", padx=10)

        # Button row
        ar = tk.Frame(p, bg=BG); ar.pack(anchor="w", pady=(0, 10))
        self._gbtn  = mkbtn(ar, "⚡  GENERATE QUESTIONS", self._do_gen,
                             padx=26, pady=10, font=("Courier New", 12, "bold"))
        self._gbtn.pack(side="left", padx=(0, 16))
        self._gprog = tk.Label(ar, text="", bg=BG, fg=MUTED, font=FS)
        self._gprog.pack(side="left")

        # Two-panel output
        cols = tk.Frame(p, bg=BG); cols.pack(fill="both", expand=True)

        # Left — generated output
        lf = tk.Frame(cols, bg=BG); lf.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(lf, text="OUTPUT", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        ow = tk.Frame(lf, bg=BORDER, bd=1); ow.pack(fill="both", expand=True)
        self._gout = scrolledtext.ScrolledText(
            ow, bg=PANEL, fg=TEXT, font=FM, relief="flat",
            padx=14, pady=12, wrap=tk.WORD, insertbackground=ACCENT, state="disabled"
        )
        self._gout.pack(fill="both", expand=True)

        # Right — retrieved context
        rf = tk.Frame(cols, bg=BG, width=320); rf.pack(side="right", fill="y", padx=(8, 0))
        rf.pack_propagate(False)
        tk.Label(rf, text="RETRIEVED CONTEXT", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        cw = tk.Frame(rf, bg=BORDER, bd=1); cw.pack(fill="both", expand=True)
        self._cout = scrolledtext.ScrolledText(
            cw, bg=PANEL2, fg=MUTED, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, state="disabled"
        )
        self._cout.pack(fill="both", expand=True)

        # Bottom actions
        bot = tk.Frame(p, bg=BG); bot.pack(fill="x", pady=(10, 0))
        self._savebtn = mkbtn(bot, "✓  Save to Knowledge Base", self._save_kb,
                               bg=SUCCESS, pady=6, padx=14, state="disabled")
        self._savebtn.pack(side="left", padx=(0, 10))
        self._copybtn = mkbtn(bot, "⊞  Copy", self._copy,
                               bg=PANEL, fg=TEXT, font=FM, pady=6, padx=14, state="disabled")
        self._copybtn.pack(side="left")

    # ── Knowledge Base tab ────────────────────────────────────────────────────

    def _tab_kb(self):
        top = tk.Frame(self._t_kb, bg=BG); top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Knowledge Base", bg=BG, fg=TEXT, font=FT).pack(side="left")

        fr = tk.Frame(top, bg=BG); fr.pack(side="right")
        tk.Label(fr, text="Filter:", bg=BG, fg=MUTED, font=FS).pack(side="left", padx=(0, 8))
        self._kbf = tk.StringVar(value="ALL")
        for v in ["ALL"] + list(SECTIONS.keys()):
            tk.Radiobutton(fr, text=v, variable=self._kbf, value=v,
                           bg=BG, fg=MUTED, selectcolor=PANEL, activebackground=BG,
                           font=FS, cursor="hand2", command=self._refresh_kb
                           ).pack(side="left", padx=3)

        tf = tk.Frame(self._t_kb, bg=BG); tf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        cols = ("ID", "Sec", "Source", "Qs", "Indexed", "Date")
        self._kbt = ttk.Treeview(tf, columns=cols, show="headings", height=14)
        for c, w in zip(cols, (44, 54, 100, 40, 74, 150)):
            self._kbt.heading(c, text=c)
            self._kbt.column(c, width=w, anchor="w" if c == "Date" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._kbt.yview)
        self._kbt.configure(yscrollcommand=vsb.set)
        self._kbt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._kbt.bind("<<TreeviewSelect>>", self._kb_sel)

        pf = tk.Frame(self._t_kb, bg=BORDER); pf.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        self._kbprev = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat", padx=12, pady=10, wrap=tk.WORD, height=10
        )
        self._kbprev.pack(fill="both", expand=True)
        self._refresh_kb()

    # ── Output History tab ────────────────────────────────────────────────────

    def _tab_out(self):
        top = tk.Frame(self._t_out, bg=BG); top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Output History", bg=BG, fg=TEXT, font=FT).pack(side="left")
        mkbtn(top, "↻  Refresh", self._refresh_out, bg=PANEL, fg=MUTED, font=FS, pady=5
              ).pack(side="right")

        tf = tk.Frame(self._t_out, bg=BG); tf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        cols = ("ID", "Section", "Context Docs", "Generated At")
        self._outt = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (50, 180, 110, 160)):
            self._outt.heading(c, text=c)
            self._outt.column(c, width=w, anchor="w" if c == "Generated At" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._outt.yview)
        self._outt.configure(yscrollcommand=vsb.set)
        self._outt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._outt.bind("<<TreeviewSelect>>", self._out_sel)

        pf = tk.Frame(self._t_out, bg=BORDER); pf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        self._outprev = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat", padx=12, pady=10, wrap=tk.WORD
        )
        self._outprev.pack(fill="both", expand=True)

        bot = tk.Frame(self._t_out, bg=BG); bot.pack(fill="x", padx=22, pady=(0, 10))
        self._promobtn = mkbtn(bot, "✓  Add to Knowledge Base", self._promote,
                                bg=SUCCESS, pady=6, padx=14, state="disabled")
        self._promobtn.pack(side="left")
        self._refresh_out()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _chk_ollama(self):
        ok = Ollama.is_up()
        if ok:
            self._dot.config(fg=SUCCESS); self._dlbl.config(text="Ollama connected ✓", fg=SUCCESS)
            self._refresh_models()
        else:
            self._dot.config(fg=DANGER); self._dlbl.config(text="Ollama offline", fg=DANGER)
            self._status("⚠  Ollama offline — install at https://ollama.ai  |  "
                         "ollama pull llama3.2  &&  ollama pull nomic-embed-text")

    def _refresh_models(self):
        models = Ollama.list_models()
        if not models: return
        self._llm_cb["values"] = models
        self._emb_cb["values"] = models
        if self._llm_var.get() not in models: self._llm_var.set(models[0])
        pref = next((m for m in models if "embed" in m.lower()), models[0])
        if self._emb_var.get() not in models: self._emb_var.set(pref)

    def _do_index(self):
        self._idx_btn.config(state="disabled", text="Indexing…")
        self.rag.emb = self._emb_var.get()

        def worker():
            total = len(self.db.get_unindexed())
            if total == 0:
                self.after(0, lambda: (
                    self._idx_lbl.config(text="Nothing to index"),
                    self._idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self._status("All documents already indexed")
                ))
                return
            try:
                def prog(i, n):
                    self.after(0, lambda _i=i, _n=n: (
                        self._idx_lbl.config(text=f"{_i+1}/{_n}…"),
                        self._refresh_stats()
                    ))
                done = self.rag.index_all(on_progress=prog)
                self.after(0, lambda: (
                    self._idx_lbl.config(text=f"✓ {done} indexed"),
                    self._idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self._refresh_stats(), self._refresh_kb(),
                    self._status(f"Indexed {done} document(s)")
                ))
            except Exception as e:
                self.after(0, lambda err=str(e): (
                    self._idx_lbl.config(text="Error"),
                    self._idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self._status(f"Index error: {err[:100]}")
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _do_gen(self):
        if not Ollama.is_up():
            messagebox.showerror("Ollama Not Running",
                "Install Ollama from https://ollama.ai\n\n"
                "Then run in terminal:\n"
                "  ollama pull llama3.2\n"
                "  ollama pull nomic-embed-text")
            return

        section = self._gsec.get()
        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Add documents → click 'Index Knowledge Base'.\n\n"
                "Generate anyway (no RAG context)?"):
                return

        self.rag.llm = self._llm_var.get()
        self.rag.emb = self._emb_var.get()
        hint = self._hint.get()

        self._gbtn.config(state="disabled", text="Generating…")
        self._savebtn.config(state="disabled"); self._copybtn.config(state="disabled")
        self._gprog.config(text="")
        self._wout("Contacting local AI model…\n", self._gout)
        self._wout("", self._cout)

        def worker():
            try:
                data, retrieved = self.rag.generate(
                    section, hint,
                    on_progress=lambda m: self.after(0, lambda msg=m: self._gprog.config(text=f"⟳  {msg}"))
                )
                self.after(0, lambda: self._gen_ok(data, retrieved, section))
            except Exception as e:
                self.after(0, lambda err=str(e): self._gen_err(err))

        threading.Thread(target=worker, daemon=True).start()

    def _gen_ok(self, data, retrieved, section):
        self._last_data = data; self._last_section = section; self._last_retrieved = retrieved
        self._wout(format_qset(data), self._gout)

        ctx = f"Retrieved {len(retrieved)} document(s):\n\n" if retrieved else "No indexed docs retrieved.\n"
        for i, (sc, doc) in enumerate(retrieved, 1):
            ctx += (f"[{i}] ID #{doc['id']}  ·  {doc['section']}  "
                    f"·  sim {sc:.3f}  ·  {doc['source']}\n"
                    f"    {doc['embed_text'][:110].replace(chr(10),' ')}…\n\n")
        self._wout(ctx.strip(), self._cout)

        self._gbtn.config(state="normal", text="⚡  GENERATE QUESTIONS")
        self._savebtn.config(state="normal"); self._copybtn.config(state="normal")
        self._gprog.config(text="✓  Done!")
        self._refresh_stats(); self._refresh_out()
        self._status(f"Generated {SECTIONS[section]} using {len(retrieved)} context doc(s)")

    def _gen_err(self, err):
        self._wout(f"ERROR\n{'─'*40}\n{err}", self._gout)
        self._gbtn.config(state="normal", text="⚡  GENERATE QUESTIONS")
        self._gprog.config(text="")
        self._status(f"Error: {err[:80]}")

    def _wout(self, text, widget):
        widget.config(state="normal"); widget.delete(1.0, tk.END)
        widget.insert(tk.END, text); widget.config(state="disabled")

    def _save_kb(self):
        if self._last_data:
            self.db.add_doc(self._last_section, self._last_data, source="generated")
            self._refresh_stats(); self._refresh_kb()
            self._gprog.config(text="✓  Saved — re-index to activate for RAG")
            self._savebtn.config(state="disabled")

    def _copy(self):
        self._gout.config(state="normal")
        t = self._gout.get(1.0, tk.END)
        self._gout.config(state="disabled")
        self.clipboard_clear(); self.clipboard_append(t)
        self._copybtn.config(text="✓  Copied!")
        self.after(2000, lambda: self._copybtn.config(text="⊞  Copy"))

    def _import(self):
        p = filedialog.askopenfilename(
            title="Import UCAT Questions (JSON)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if not p: return
        try:
            n = self.db.import_json(p)
            messagebox.showinfo("Import", f"Imported {n} document(s).\nNow click '⊛ Index Knowledge Base'.")
            self._refresh_stats(); self._refresh_kb()
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def _add_samples(self):
        for q in SAMPLES:
            self.db.add_doc(q["section"], q, source="sample")
        messagebox.showinfo("Samples Added",
            f"Added {len(SAMPLES)} sample documents.\nNow click '⊛ Index Knowledge Base' to embed them.")
        self._refresh_stats(); self._refresh_kb()

    def _export(self):
        rows = self.db.get_generated(limit=10000)
        if not rows:
            messagebox.showinfo("Export", "No generated questions yet."); return
        p = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="ucat_generated.json"
        )
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump([r["data"] for r in rows], f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Exported", f"Saved {len(rows)} sets to:\n{p}")

    def _refresh_kb(self):
        filt = self._kbf.get()
        docs = self.db.get_all_docs(None if filt == "ALL" else filt, limit=2000)
        for iid in self._kbt.get_children(): self._kbt.delete(iid)
        for d in docs:
            self._kbt.insert("", "end", iid=str(d["id"]),
                              values=(d["id"], d["section"], d["source"],
                                      len(d["data"].get("questions", [])),
                                      "✓" if d["embedding"] else "○",
                                      d["created"][:16]))

    def _kb_sel(self, _e):
        sel = self._kbt.selection()
        if not sel: return
        docs = self.db.get_all_docs(limit=100000)
        doc  = next((d for d in docs if d["id"] == int(sel[0])), None)
        if doc:
            self._kbprev.config(state="normal")
            self._kbprev.delete(1.0, tk.END)
            self._kbprev.insert(tk.END, format_qset(doc["data"]))
            self._kbprev.config(state="disabled")

    def _refresh_out(self):
        rows = self.db.get_generated(limit=500)
        for iid in self._outt.get_children(): self._outt.delete(iid)
        for r in rows:
            self._outt.insert("", "end", iid=str(r["id"]),
                               values=(r["id"], SECTIONS.get(r["section"], r["section"]),
                                       len(r["context_ids"]), r["created"][:16]))

    def _out_sel(self, _e):
        sel = self._outt.selection()
        if not sel: return
        self._sel_gen_id = int(sel[0])
        rows = self.db.get_generated(limit=500)
        row  = next((r for r in rows if r["id"] == self._sel_gen_id), None)
        if row:
            self._outprev.config(state="normal")
            self._outprev.delete(1.0, tk.END)
            self._outprev.insert(tk.END, format_qset(row["data"]))
            self._outprev.config(state="disabled")
            self._promobtn.config(state="normal")

    def _promote(self):
        if self._sel_gen_id:
            self.db.promote_to_kb(self._sel_gen_id)
            self._refresh_stats(); self._refresh_kb()
            self._promobtn.config(state="disabled")
            self._status("Added to knowledge base — re-index to activate for RAG")

    def _refresh_stats(self):
        for code in SECTIONS:
            idx = self.db.count(code, indexed_only=True)
            tot = self.db.count(code, indexed_only=False)
            self._slabels[code].config(text=f"{idx}/{tot}")

    def _status(self, msg):
        self._sbar.config(text=msg)
        self.after(7000, lambda: self._sbar.config(text="Ready"))

    def on_close(self):
        self.db.close(); self.destroy()


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
