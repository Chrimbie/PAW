"""Minimal PAW, optimized build (2026-10-30).

8-parameter core (1 input, 2 hidden, 1 output) + ReflexStore. Offload
on activation, pull by task key. See README.md.
"""

from .core import PlainCore
from .engine import MinimalPAW
from .store import ChunkStore

__all__ = ["PlainCore", "MinimalPAW", "ChunkStore"]
