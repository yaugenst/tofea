"""Linear system solver abstractions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial
from typing import Any

from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


class Solver(ABC):
    """Abstract interface for linear solvers."""

    @abstractmethod
    def factor(self, m: csc_matrix) -> None: ...

    @abstractmethod
    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray: ...

    @abstractmethod
    def clear(self) -> None: ...


class SuperLU(Solver):
    """`scipy.sparse.linalg.splu` wrapper."""

    def __init__(
        self, **options: float | int | bool | str | Mapping[str, bool]
    ) -> None:
        """Create a new ``SuperLU`` solver instance."""
        # store solver-specific context on the instance to avoid cross-talk
        self._ctx: dict[str, Any] = {"splu": partial(splu, **options)}

    def factor(self, m: csc_matrix) -> None:
        """Compute the factorization of ``m``."""
        self._ctx["factorization"] = self._ctx["splu"](m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        """Solve the linear system."""
        return self._ctx["factorization"].solve(rhs, trans="T" if transpose else "N")

    def clear(self) -> None:
        """Remove cached factorization."""
        self._ctx.pop("factorization", None)


def get_solver(solver: str) -> Solver:
    """Return a solver instance by name.

    Parameters
    ----------
    solver
        Name of the solver implementation. Currently only ``"SuperLU"`` is
        available.

    Examples
    --------
    >>> from tofea.solvers import get_solver, Solver
    >>> s = get_solver("SuperLU")
    >>> isinstance(s, Solver)
    True
    """
    match solver:
        case "SuperLU":
            return SuperLU(
                diag_pivot_thresh=0.1,
                permc_spec="MMD_AT_PLUS_A",
                options={"SymmetricMode": True},
            )
        case _:
            raise ValueError(f"Invalid solver: {solver}")
