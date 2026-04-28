"""Built-in sample questions in the new structured format.

These are seeded into the KB to teach Claude the expected shape — including
structured QR charts, AR shape panels, and DM Venn specs.
"""
from __future__ import annotations

from typing import List, Dict, Any

# Coverage tag helpers — keeps each question dict short.
def _cov(topic: str, scenario: str, named: bool = False, ctx: str = "UK") -> Dict[str, Any]:
    return {
        "topic": topic,
        "scenario_type": scenario,
        "contains_named_entities": named,
        "cultural_context": ctx,
    }


SAMPLES: List[Dict[str, Any]] = [
    # ────────────────────────────── VR ──────────────────────────────
    {
        "section": "VR",
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
            {"number": 1, "text": "Air conditioning systems contribute to the UHI effect.",
             "type": "tf", "options": {"A": "True", "B": "False", "C": "Can't Tell"},
             "answer": "A", "difficulty": 1.5,
             "explanation": "The passage explicitly lists waste heat from air conditioning as a contributing factor.",
             "coverage": _cov("urban planning", "scientific")},
            {"number": 2, "text": "Urban areas are only warmer than rural areas during the daytime.",
             "type": "tf", "options": {"A": "True", "B": "False", "C": "Can't Tell"},
             "answer": "B", "difficulty": 2.5,
             "explanation": "Materials re-radiate heat overnight, so the UHI effect persists at night.",
             "coverage": _cov("urban planning", "scientific")},
            {"number": 3, "text": "Which best describes the primary cause of the UHI effect?",
             "type": "mc",
             "options": {"A": "Industrial air pollution trapping heat",
                         "B": "Replacement of natural surfaces with heat-absorbing materials",
                         "C": "Increased vehicle exhaust in city centres",
                         "D": "Deforestation in rural areas surrounding cities"},
             "answer": "B", "difficulty": 2.0,
             "explanation": "The passage identifies this replacement as the primary cause in its opening sentences.",
             "coverage": _cov("urban planning", "scientific")},
            {"number": 4, "text": "Doubling tree canopy cover guarantees an 8°C temperature reduction.",
             "type": "tf", "options": {"A": "True", "B": "False", "C": "Can't Tell"},
             "answer": "B", "difficulty": 3.0,
             "explanation": "The passage says 'between 2°C and 8°C depending on local conditions' — 8°C is a maximum, not guaranteed.",
             "coverage": _cov("urban planning", "scientific")},
        ],
    },

    # ────────────────────────────── DM ──────────────────────────────
    {
        "section": "DM",
        "questions": [
            {"number": 1, "type": "syllogism",
             "text": "Premise 1: All surgeons are doctors.\nPremise 2: Some doctors work night shifts.\n\nWhich conclusion follows?",
             "options": {"A": "All surgeons work night shifts", "B": "Some surgeons work night shifts",
                          "C": "No surgeons work night shifts", "D": "Some doctors are not surgeons",
                          "E": "None of the above"},
             "answer": "E", "difficulty": 3.0,
             "explanation": "The 'some doctors' who work nights may or may not overlap with surgeons. No conclusion about surgeons can be drawn.",
             "coverage": _cov("logic", "abstract")},
            {"number": 2, "type": "logical",
             "text": ("Five friends — Alex, Ben, Cara, Dan, Eve — each own one pet: cat, dog, rabbit, hamster, fish.\n"
                      "• Alex does not own a cat or dog.\n• Ben owns the rabbit.\n"
                      "• Cara owns the fish.\n• Dan does not own the hamster.\n"
                      "Which must be true?"),
             "options": {"A": "Alex owns the hamster", "B": "Dan owns the cat",
                          "C": "Eve owns the dog", "D": "Alex owns the fish", "E": "Eve owns the hamster"},
             "answer": "A", "difficulty": 3.5,
             "explanation": "Ben=rabbit, Cara=fish. Alex cannot have cat/dog/rabbit/fish, so Alex=hamster.",
             "coverage": _cov("logic puzzles", "everyday", named=True)},
            {"number": 3, "type": "probability",
             "text": "A bag contains 6 red, 4 blue, and 2 yellow counters. One drawn at random. P(NOT red)?",
             "options": {"A": "1/6", "B": "1/3", "C": "1/2", "D": "2/3", "E": "5/6"},
             "answer": "C", "difficulty": 2.0,
             "explanation": "Non-red = 4+2 = 6. Total = 12. P = 6/12 = 1/2.",
             "coverage": _cov("probability", "abstract")},
            {"number": 4, "type": "argument",
             "text": "Statement: The voting age should be lowered to 16.\nStrongest argument IN FAVOUR?",
             "options": {
                 "A": "Teenagers are affected by government policy and pay tax, yet have no democratic voice",
                 "B": "Some 16-year-olds are more mature than some adults",
                 "C": "Other countries have adopted this policy",
                 "D": "It would increase overall voter turnout",
                 "E": "Young people are more engaged with social media"},
             "answer": "A", "difficulty": 3.5,
             "explanation": "A directly links democratic rights to civic obligations already borne by 16-year-olds.",
             "coverage": _cov("political ethics", "social")},
            {"number": 5, "type": "venn",
             "text": ("All chefs are creative. Some creative people are musicians. No musicians are accountants.\n"
                      "Which must be true?"),
             "options": {"A": "Some chefs are musicians", "B": "No chefs are accountants",
                          "C": "No accountants are creative", "D": "Some creative people are not accountants",
                          "E": "All musicians are chefs"},
             "answer": "D", "difficulty": 4.0,
             "explanation": "Creative musicians exist (given), and no musicians are accountants, so those creative-musicians are not accountants.",
             "coverage": _cov("set theory", "abstract"),
             "venn": {
                 "universe_label": "Creative people, musicians, and accountants",
                 "sets": [
                     {"label": "Creative", "members": ["c1", "c2", "c3", "c4", "m1", "m2", "ch1", "ch2"]},
                     {"label": "Musicians", "members": ["m1", "m2", "m3"]},
                     {"label": "Accountants", "members": ["a1", "a2"]},
                 ],
             }},
        ],
    },

    # ────────────────────────────── QR ──────────────────────────────
    {
        "section": "QR",
        "stimulus": {
            "type": "bar",
            "title": "Monthly Sales — TechStore UK",
            "x_label": "Month",
            "y_label": "Sales",
            "units": "£000s",
            "categories": ["Jan", "Feb", "Mar", "Apr"],
            "series": [
                {"name": "Laptops",     "values": [84, 91, 107, 98]},
                {"name": "Phones",      "values": [62, 58, 74, 89]},
                {"name": "Accessories", "values": [23, 27, 31, 28]},
                {"name": "Gaming",      "values": [45, 39, 52, 61]},
            ],
        },
        "questions": [
            {"number": 1, "text": "What is the mean monthly total sales across the four months?",
             "options": {"A": "£232,250", "B": "£238,000", "C": "£242,250", "D": "£244,000", "E": "£248,500"},
             "answer": "C", "difficulty": 2.5,
             "explanation": "(214+215+264+276)/4 = 969/4 = 242.25 → £242,250",
             "coverage": _cov("retail", "business")},
            {"number": 2, "text": "Phones sales grew by what percentage from February to April?",
             "options": {"A": "43.5%", "B": "48.3%", "C": "51.2%", "D": "53.4%", "E": "56.7%"},
             "answer": "D", "difficulty": 3.0,
             "explanation": "(89−58)/58 × 100 = 31/58 × 100 = 53.4%",
             "coverage": _cov("retail", "business")},
            {"number": 3, "text": "In March, what fraction of total sales were Accessories?",
             "options": {"A": "31/264", "B": "1/9", "C": "1/8", "D": "7/56", "E": "1/7"},
             "answer": "A", "difficulty": 3.5,
             "explanation": "31/264 — this does not simplify to any other option.",
             "coverage": _cov("retail", "business")},
            {"number": 4, "text": "If Gaming grows 20% from April, what will May Gaming sales be?",
             "options": {"A": "£68,200", "B": "£70,800", "C": "£71,400", "D": "£73,200", "E": "£75,600"},
             "answer": "D", "difficulty": 2.5,
             "explanation": "61 × 1.20 = 73.2 → £73,200",
             "coverage": _cov("retail", "business")},
        ],
    },

    # ────────────────────────────── AR ──────────────────────────────
    {
        "section": "AR",
        "set_a_rule": (
            "Each panel contains at least one black square AND at least one white circle. "
            "No other shape kinds appear."
        ),
        "set_a_panels": [
            {"label": "Panel 1", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Panel 2", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Panel 3", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "square", "color": "black"},
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Panel 4", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Panel 5", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Panel 6", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "square", "color": "black"},
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"},
                {"kind": "circle", "color": "white"}]},
        ],
        "set_b_rule": "Each panel contains ONLY triangles — any number, no other kinds.",
        "set_b_panels": [
            {"label": "Panel 1", "shapes": [{"kind": "triangle", "color": "black"}] * 3},
            {"label": "Panel 2", "shapes": [{"kind": "triangle", "color": "black"}] * 2},
            {"label": "Panel 3", "shapes": [{"kind": "triangle", "color": "black"}] * 4},
            {"label": "Panel 4", "shapes": [{"kind": "triangle", "color": "black"}]},
            {"label": "Panel 5", "shapes": [{"kind": "triangle", "color": "black"}] * 5},
            {"label": "Panel 6", "shapes": [{"kind": "triangle", "color": "black"}] * 6},
        ],
        "test_panels": [
            {"label": "Test 1", "shapes": [{"kind": "square", "color": "black"}] * 3},
            {"label": "Test 2", "shapes": [{"kind": "triangle", "color": "black"}] * 2},
            {"label": "Test 3", "shapes": [
                {"kind": "square", "color": "black"},
                {"kind": "circle", "color": "white"},
                {"kind": "circle", "color": "white"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Test 4", "shapes": [
                {"kind": "triangle", "color": "black"},
                {"kind": "circle", "color": "white"}]},
            {"label": "Test 5", "shapes": [
                {"kind": "square", "color": "white"},
                {"kind": "circle", "color": "white"}]},
        ],
        "questions": [
            {"number": 1, "text": "Test 1 (3 black squares, no circles)",
             "options": {"A": "Set A", "B": "Set B", "C": "Neither"},
             "answer": "C", "difficulty": 2.5,
             "explanation": "Has squares but no circle — fails Set A's requirement for at least one circle.",
             "coverage": _cov("pattern recognition", "abstract")},
            {"number": 2, "text": "Test 2 (2 black triangles only)",
             "options": {"A": "Set A", "B": "Set B", "C": "Neither"},
             "answer": "B", "difficulty": 1.5,
             "explanation": "Contains only triangles — matches Set B's rule.",
             "coverage": _cov("pattern recognition", "abstract")},
            {"number": 3, "text": "Test 3 (1 black square and 3 white circles)",
             "options": {"A": "Set A", "B": "Set B", "C": "Neither"},
             "answer": "A", "difficulty": 2.0,
             "explanation": "Contains at least one square and at least one circle — matches Set A.",
             "coverage": _cov("pattern recognition", "abstract")},
            {"number": 4, "text": "Test 4 (1 triangle and 1 circle)",
             "options": {"A": "Set A", "B": "Set B", "C": "Neither"},
             "answer": "C", "difficulty": 3.0,
             "explanation": "No square (fails Set A). Not only triangles (circle present, fails Set B).",
             "coverage": _cov("pattern recognition", "abstract")},
            {"number": 5, "text": "Test 5 (1 white square and 1 white circle)",
             "options": {"A": "Set A", "B": "Set B", "C": "Neither"},
             "answer": "C", "difficulty": 3.5,
             "explanation": "Square is white not black — Set A requires black squares.",
             "coverage": _cov("pattern recognition", "abstract")},
        ],
    },
]
