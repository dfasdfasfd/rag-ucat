#!/usr/bin/env python3
"""
UCAT Trainer · RAG Edition v2
Best-in-field question generation grounded in your knowledge base.
Requires: Python 3.8+ · Ollama (https://ollama.ai)

Models to pull:
  ollama pull mxbai-embed-large
  ollama pull qwen2.5:14b        (or qwen2.5:7b for low-RAM)
  ollama pull qwen2.5:7b         (quality scorer)
  ollama pull qwen2.5vl          (screenshot OCR — optional)
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui.app import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
