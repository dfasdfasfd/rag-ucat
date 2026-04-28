#!/usr/bin/env python3
"""UCAT Trainer · entry-point shim.

The implementation has been modularized into the ``ucat`` package. This file
remains as a stable entry point (`python ucat_trainer.py`) — the equivalent
modular invocation is ``python -m ucat``.
"""
from ucat.ui import run


if __name__ == "__main__":
    run()
