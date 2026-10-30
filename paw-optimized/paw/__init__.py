"""PAW, optimized build (2026-10-30).

MLP + ReflexStore: weights as chunks, offload on activation, pull when
needed. See README.md for the architecture, the optimizations, and what
is deliberately not built yet.
"""

from .engine import PAW
from .mlp import MLP
from .router import TaskRouter
from .store import ChunkStore

__all__ = ["PAW", "MLP", "TaskRouter", "ChunkStore"]
