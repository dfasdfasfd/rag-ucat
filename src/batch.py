"""
Batch generation: background thread execution, difficulty distribution,
multi-format export (JSON, CSV, PDF with runtime guard), and failure logging.
"""

import json
import csv
import io

from src.config import (
    DIFFICULTY_MAP, HAS_REPORTLAB, QUALITY_THRESHOLDS, SECTIONS,
)
from src.embeddings import EmbeddingEngine


class BatchGenerator:
    """
    Batch question generation, unlocked per-section after calibration.
    Generates with difficulty distribution, quality gates, and export.
    """

    def __init__(self, generator, calibration_manager, db):
        self.generator = generator
        self.calibration = calibration_manager
        self.db = db

    def generate_batch(self, section: str, count: int,
                       difficulty_dist: dict,
                       output_format: str = "json",
                       on_progress=None) -> dict:
        """
        Generate a batch of questions for a calibrated section.

        Args:
            section: VR/DM/QR/SJT
            count: total question sets to generate
            difficulty_dist: {"easy": 0.3, "medium": 0.4, "hard": 0.3}
            output_format: "json" | "csv" | "pdf"
            on_progress: callback(successful, total, failed)

        Returns: {successful, failed, failures, output}
        """
        if not self.calibration.is_batch_unlocked(section):
            raise RuntimeError(
                f"{SECTIONS[section]} has not passed calibration yet. "
                f"Complete calibration first."
            )

        if output_format == "pdf" and not HAS_REPORTLAB:
            raise RuntimeError(
                "PDF export requires reportlab. Install with: pip install reportlab"
            )

        results = []
        failures = []

        # Compute how many at each difficulty
        targets = {}
        for diff_name, pct in difficulty_dist.items():
            targets[diff_name] = round(count * pct)

        # Adjust rounding to match total
        total_target = sum(targets.values())
        if total_target < count:
            # Add remainder to medium
            targets["medium"] = targets.get("medium", 0) + (count - total_target)
        elif total_target > count:
            targets["medium"] = max(0, targets.get("medium", 0) - (total_target - count))

        for diff_name, target_count in targets.items():
            diff_value = DIFFICULTY_MAP.get(diff_name, 5)

            for i in range(target_count):
                if on_progress:
                    on_progress(len(results), count, len(failures))

                try:
                    data, retrieved, report = self.generator.generate(
                        section, difficulty=diff_value
                    )

                    # Quality gate
                    if (report.get("rule_score", 0) >= QUALITY_THRESHOLDS["auto_promote_rule"]
                            and report.get("llm_score", 0) >= QUALITY_THRESHOLDS["auto_promote_llm"]):
                        results.append({
                            "data": data,
                            "difficulty": diff_name,
                            "quality": report,
                        })
                        # Auto-promote to KB
                        embed_text = EmbeddingEngine.embed_text_for(data, section)
                        self.db.promote_data_to_kb(
                            section, data, embed_text, source="batch"
                        )
                    else:
                        failures.append({
                            "data": data,
                            "difficulty": diff_name,
                            "report": report,
                            "reason": (
                                f"Below quality threshold "
                                f"(rule: {report.get('rule_score', 0):.2f}, "
                                f"llm: {report.get('llm_score', 0):.1f})"
                            ),
                        })

                except RuntimeError as e:
                    failures.append({
                        "error": str(e),
                        "difficulty": diff_name,
                        "reason": "Generation failed after retries",
                    })

        # Final progress
        if on_progress:
            on_progress(len(results), count, len(failures))

        # Export
        output_data = [r["data"] for r in results]
        output = self._export(output_data, output_format, section)

        return {
            "successful": len(results),
            "failed": len(failures),
            "total_attempted": count,
            "failures": failures,
            "output": output,
            "format": output_format,
        }

    # ─── Export Formats ──────────────────────────────────────────────────────

    def _export(self, results: list, fmt: str, section: str) -> str:
        if fmt == "json":
            return self._to_json(results)
        elif fmt == "csv":
            return self._to_csv(results, section)
        elif fmt == "pdf":
            return self._to_pdf(results, section)
        return self._to_json(results)

    def _to_json(self, results: list) -> str:
        return json.dumps(results, indent=2, ensure_ascii=False)

    def _to_csv(self, results: list, section: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output)

        if section == "VR":
            writer.writerow(["passage", "q_num", "q_text", "q_type",
                             "opt_a", "opt_b", "opt_c", "opt_d",
                             "answer", "explanation"])
            for data in results:
                passage = data.get("passage", "")
                for q in data.get("questions", []):
                    opts = q.get("options", {})
                    writer.writerow([
                        passage, q.get("number"), q.get("text"), q.get("type", ""),
                        opts.get("A", ""), opts.get("B", ""),
                        opts.get("C", ""), opts.get("D", ""),
                        q.get("answer", ""), q.get("explanation", ""),
                    ])

        elif section == "DM":
            writer.writerow(["q_num", "q_type", "q_text",
                             "opt_a", "opt_b", "opt_c", "opt_d", "opt_e",
                             "answer", "explanation"])
            for data in results:
                for q in data.get("questions", []):
                    opts = q.get("options", {})
                    writer.writerow([
                        q.get("number"), q.get("type", ""), q.get("text"),
                        opts.get("A", ""), opts.get("B", ""),
                        opts.get("C", ""), opts.get("D", ""), opts.get("E", ""),
                        q.get("answer", ""), q.get("explanation", ""),
                    ])

        elif section == "QR":
            writer.writerow(["stimulus", "q_num", "q_text",
                             "opt_a", "opt_b", "opt_c", "opt_d", "opt_e",
                             "answer", "explanation"])
            for data in results:
                stimulus = data.get("stimulus", "")
                for q in data.get("questions", []):
                    opts = q.get("options", {})
                    writer.writerow([
                        stimulus, q.get("number"), q.get("text"),
                        opts.get("A", ""), opts.get("B", ""),
                        opts.get("C", ""), opts.get("D", ""), opts.get("E", ""),
                        q.get("answer", ""), q.get("explanation", ""),
                    ])

        elif section == "SJT":
            writer.writerow(["scenario", "q_num", "q_text",
                             "opt_a", "opt_b", "opt_c", "opt_d",
                             "answer", "explanation"])
            for data in results:
                scenario = data.get("scenario", "")
                for q in data.get("questions", []):
                    opts = q.get("options", {})
                    writer.writerow([
                        scenario, q.get("number"), q.get("text"),
                        opts.get("A", ""), opts.get("B", ""),
                        opts.get("C", ""), opts.get("D", ""),
                        q.get("answer", ""), q.get("explanation", ""),
                    ])

        return output.getvalue()

    def _to_pdf(self, results: list, section: str) -> bytes:
        """Generate PDF using reportlab. Returns PDF bytes."""
        if not HAS_REPORTLAB:
            raise RuntimeError("PDF export requires reportlab")

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import mm

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(
            f"UCAT {SECTIONS[section]} — Generated Questions",
            styles["Title"]
        ))
        story.append(Spacer(1, 10 * mm))

        for idx, data in enumerate(results):
            story.append(Paragraph(
                f"Question Set {idx + 1}",
                styles["Heading2"]
            ))

            # Passage / Stimulus
            for key in ("passage", "stimulus", "set_a_description", "set_b_description"):
                if key in data:
                    label = key.replace("_", " ").title()
                    story.append(Paragraph(f"<b>{label}:</b>", styles["Normal"]))
                    story.append(Paragraph(str(data[key]), styles["Normal"]))
                    story.append(Spacer(1, 3 * mm))

            # Questions
            for q in data.get("questions", []):
                q_text = f"Q{q.get('number', '?')}. {q.get('text', '')}"
                story.append(Paragraph(q_text, styles["Normal"]))
                for opt_key, opt_val in q.get("options", {}).items():
                    story.append(Paragraph(
                        f"&nbsp;&nbsp;&nbsp;{opt_key}) {opt_val}",
                        styles["Normal"]
                    ))
                story.append(Paragraph(
                    f"<b>Answer: {q.get('answer', '?')}</b>",
                    styles["Normal"]
                ))
                if q.get("explanation"):
                    story.append(Paragraph(
                        f"<i>{q['explanation']}</i>",
                        styles["Normal"]
                    ))
                story.append(Spacer(1, 3 * mm))

            story.append(Spacer(1, 8 * mm))

        doc.build(story)
        return buffer.getvalue()
