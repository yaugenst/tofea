"""Light-weight, differentiable finite element analysis package."""

from __future__ import annotations

__all__ = [
    "DEFAULT_SOLVER",
    "__version__",
    "boundary_conditions",
    "elements",
    "fea2d",
    "primitives",
    "solvers",
]

__version__ = "0.1.0"

DEFAULT_SOLVER = "SuperLU"
