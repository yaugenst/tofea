"""Light-weight, differentiable finite element analysis package."""

from __future__ import annotations

__all__ = [
    "elements",
    "fea2d",
    "primitives",
    "solvers",
    "DEFAULT_SOLVER",
    "__version__",
]

__version__ = "0.1.0"

DEFAULT_SOLVER = "SuperLU"
