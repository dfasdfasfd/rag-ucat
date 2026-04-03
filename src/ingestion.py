"""
Screenshot OCR ingestion via Ollama vision models.
Multi-pass validation + cleanup for handling OCR noise from
watermarks, timers, and navigation elements.
"""

import json

from src.config import SECTIONS, SECTION_SCHEMAS, DEFAULT_VISION
from src.quality import validate_schema


class ScreenshotIngester:
    """Extract structured question data from UCAT screenshots via vision OCR."""

    MAX_CLEANUP_PASSES = 3

    def __init__(self, ollama_client, db, embedding_engine,
                 vision_model: str = DEFAULT_VISION,
                 cleanup_model: str = "qwen2.5:14b"):
        self.ollama = ollama_client
        self.db = db
        self.embeddings = embedding_engine
        self.vision_model = vision_model
        self.cleanup_model = cleanup_model

    def ingest(self, image_path: str, section: str,
               on_progress=None) -> dict:
        """
        Extract structured question data from a UCAT screenshot.
        Uses multi-pass validation + cleanup to handle OCR noise.
        Returns the parsed data dict.
        Raises RuntimeError if cleanup fails after MAX_CLEANUP_PASSES.
        """
        if on_progress:
            on_progress("Running OCR via vision model...")

        schema = SECTION_SCHEMAS[section]

        # 1. Vision OCR with section-specific prompt
        ocr_prompt = (
            f"Extract the UCAT {SECTIONS[section]} question from this screenshot.\n"
            f"IGNORE watermarks, timers, progress bars, and navigation elements.\n"
            f"Return ONLY the question content as JSON matching this exact schema:\n"
            f"{schema['shape']}\n\n"
            f"Section description: {schema['desc']}"
        )

        raw_json = self.ollama.vision_extract(
            image_path, model=self.vision_model, prompt=ocr_prompt
        )

        # 2. Multi-pass validation + cleanup
        errors = None
        data = None

        for pass_num in range(self.MAX_CLEANUP_PASSES):
            # Try parsing
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                data = None

            if data is None:
                errors = ["Output is not valid JSON"]
            else:
                data["section"] = section
                errors = validate_schema(data, section)

            if not errors:
                break  # Clean output

            if on_progress:
                on_progress(
                    f"Cleanup pass {pass_num + 1}/{self.MAX_CLEANUP_PASSES}: "
                    f"fixing {len(errors)} error(s)..."
                )

            # Increasingly explicit schema instructions on each pass
            cleanup_prompt = (
                f"Pass {pass_num + 1}: The OCR output has these errors:\n"
                + "\n".join(f"  - {e}" for e in errors) + "\n\n"
                f"The broken output was:\n{raw_json[:3000]}\n\n"
                f"The EXACT schema required is:\n{schema['shape']}\n\n"
                f"Section rules: {schema['desc']}\n\n"
                f"Fix ALL listed errors. Ignore any watermark/timer/UI text "
                f"that leaked in. Return ONLY the corrected JSON."
            )

            raw_json = self.ollama.generate(
                "You are a JSON cleanup assistant. Fix the OCR output to match the required schema.",
                cleanup_prompt,
                self.cleanup_model,
                options={"temperature": 0.1, "num_predict": 3000}
            )

        if errors:
            raise RuntimeError(
                f"OCR cleanup failed after {self.MAX_CLEANUP_PASSES} passes.\n"
                f"Remaining errors: {'; '.join(errors)}"
            )

        # 3. Build embed text and add to KB
        embed_text = self.embeddings.embed_text_for(data, section)
        data_type = self.embeddings.infer_data_type(data, section)

        doc_id = self.db.add_doc(
            section, data, embed_text,
            source="screenshot", data_type=data_type
        )

        if on_progress:
            on_progress(f"Successfully extracted and added to KB (doc #{doc_id})")

        return data

    def ingest_batch(self, image_paths: list, section: str,
                     on_progress=None) -> dict:
        """
        Ingest multiple screenshots for the same section.
        Returns summary with successes and failures.
        """
        results = {"success": [], "failed": []}

        for i, path in enumerate(image_paths):
            if on_progress:
                on_progress(f"Processing image {i+1}/{len(image_paths)}...")
            try:
                data = self.ingest(path, section, on_progress=on_progress)
                results["success"].append({"path": path, "data": data})
            except Exception as e:
                results["failed"].append({"path": path, "error": str(e)})

        return results
