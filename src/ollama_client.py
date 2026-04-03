"""
Ollama HTTP client with format:json enforcement, dynamic num_ctx,
streaming generation, vision OCR, connection checking, and retry logic.
"""

import json
import time
import base64
import requests

from src.config import OLLAMA_BASE, CONTEXT_BUDGET


class OllamaClient:
    """Stateful Ollama client with connection management and retry."""

    def __init__(self, base_url: str = OLLAMA_BASE):
        self.base = base_url
        self._models_cache = None
        self._models_cache_time = 0

    # ─── Connection & Model Discovery ────────────────────────────────────────

    def check_connection(self, timeout: int = 3) -> bool:
        """Check if Ollama is running. Call before main window renders."""
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self, use_cache: bool = True) -> list:
        """List available Ollama models. Caches for 30 seconds."""
        now = time.time()
        if use_cache and self._models_cache and (now - self._models_cache_time) < 30:
            return self._models_cache
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=5)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                self._models_cache = models
                self._models_cache_time = now
                return models
        except Exception:
            pass
        return self._models_cache or []

    def model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        models = self.list_models()
        return any(model_name in m for m in models)

    # ─── Context Window Budget ───────────────────────────────────────────────

    def _compute_num_ctx(self, system: str, user: str, num_predict: int) -> int:
        """Compute dynamic num_ctx to prevent silent truncation."""
        total_chars = len(system) + len(user)
        estimated_input_tokens = total_chars // CONTEXT_BUDGET["chars_per_token"]
        needed = estimated_input_tokens + num_predict + 512  # safety margin
        return max(CONTEXT_BUDGET["default_ctx"], needed)

    # ─── Embedding ───────────────────────────────────────────────────────────

    def embed(self, text: str, model: str, retries: int = 2) -> list:
        """Embed text into a vector. Retries on transient failures."""
        last_err = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    f"{self.base}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60
                )
                r.raise_for_status()
                return r.json()["embedding"]
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(1 * (attempt + 1))
        raise last_err

    def embed_batch(self, texts: list, model: str, on_progress=None) -> list:
        """Embed multiple texts sequentially with progress callback."""
        results = []
        for i, text in enumerate(texts):
            vec = self.embed(text, model)
            results.append(vec)
            if on_progress:
                on_progress(i + 1, len(texts))
        return results

    # ─── Generation (non-streaming) ──────────────────────────────────────────

    def generate(self, system: str, user: str, model: str,
                 options: dict = None, retries: int = 2) -> str:
        """Generate with format:json and dynamic num_ctx. Returns raw JSON string."""
        opts = dict(options or {})
        num_predict = opts.pop("num_predict", 2800)
        num_ctx = self._compute_num_ctx(system, user, num_predict)

        payload = {
            "model": model,
            "system": system,
            "prompt": user,
            "format": "json",
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                **opts,
            },
        }

        last_err = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    f"{self.base}/api/generate",
                    json=payload,
                    timeout=300
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise last_err

    # ─── Streaming Generation ────────────────────────────────────────────────

    def generate_stream(self, system: str, user: str, model: str,
                        options: dict = None, on_token=None) -> str:
        """
        Streaming generation with format:json and dynamic num_ctx.
        Calls on_token(token_str) for each token received.
        Returns the full response string when done.
        """
        opts = dict(options or {})
        num_predict = opts.pop("num_predict", 2800)
        num_ctx = self._compute_num_ctx(system, user, num_predict)

        payload = {
            "model": model,
            "system": system,
            "prompt": user,
            "format": "json",
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                **opts,
            },
        }

        response = requests.post(
            f"{self.base}/api/generate",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            full_response += token
            if on_token and token:
                on_token(token)
            if chunk.get("done"):
                break

        return full_response

    # ─── Abort Handle ────────────────────────────────────────────────────────

    def generate_stream_abortable(self, system: str, user: str, model: str,
                                  options: dict = None, on_token=None,
                                  abort_flag=None) -> str:
        """
        Same as generate_stream but checks abort_flag() between tokens.
        abort_flag should be a callable returning True to abort.
        Returns partial response if aborted.
        """
        opts = dict(options or {})
        num_predict = opts.pop("num_predict", 2800)
        num_ctx = self._compute_num_ctx(system, user, num_predict)

        payload = {
            "model": model,
            "system": system,
            "prompt": user,
            "format": "json",
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                **opts,
            },
        }

        response = requests.post(
            f"{self.base}/api/generate",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if abort_flag and abort_flag():
                response.close()
                break
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            full_response += token
            if on_token and token:
                on_token(token)
            if chunk.get("done"):
                break

        return full_response

    # ─── Vision OCR ──────────────────────────────────────────────────────────

    def vision_extract(self, image_path: str, model: str = "qwen2.5vl",
                       prompt: str = None) -> str:
        """
        Extract structured text from an image using a vision-capable Ollama model.
        Returns raw JSON string from the model.
        """
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        if prompt is None:
            prompt = (
                "Extract all text from this image. Return the content as structured JSON."
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "format": "json",
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.1,  # Low temp for accurate extraction
            },
        }

        r = requests.post(
            f"{self.base}/api/generate",
            json=payload,
            timeout=120
        )
        r.raise_for_status()
        return r.json().get("response", "")
