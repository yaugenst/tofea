"""Linear system solver abstractions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial
from typing import Any

from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


class Solver(ABC):
    """Abstract interface for linear solvers.

    Implementations typically perform a two phase factorization of a sparse
    matrix: a symbolic analysis that depends only on the sparsity pattern and a
    numerical factorization that depends on the actual values. ``factor_full``
    is responsible for both steps while ``refactor_numerical`` reuses the
    symbolic analysis and only updates the numeric values.
    """

    @abstractmethod
    def factor_full(self, m: csc_matrix) -> None:
        """Perform a complete (symbolic and numeric) factorization of ``m``."""

    @abstractmethod
    def refactor_numerical(self, m: csc_matrix) -> None:
        """Update the numeric factorization of ``m`` using a cached symbolic analysis."""

    @abstractmethod
    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray: ...

    @abstractmethod
    def clear(self) -> None: ...


class SuperLU(Solver):
    """`scipy.sparse.linalg.splu` wrapper."""

    def __init__(self, **options: float | int | bool | str | Mapping[str, bool]) -> None:
        """Create a new ``SuperLU`` solver instance."""
        # store solver-specific context on the instance to avoid cross-talk
        self._ctx: dict[str, Any] = {"splu": partial(splu, **options)}

    def factor_full(self, m: csc_matrix) -> None:
        """Compute the complete factorization of ``m``."""
        self._ctx["factorization"] = self._ctx["splu"](m)

    def refactor_numerical(self, m: csc_matrix) -> None:
        """Update numeric values by re-factorizing ``m``.

        ``scipy``'s ``splu`` does not expose symbolic/numeric separation, so we
        simply recompute the full factorization.
        """
        self.factor_full(m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        """Solve the linear system."""
        return self._ctx["factorization"].solve(rhs, trans="T" if transpose else "N")

    def clear(self) -> None:
        """Remove cached factorization and sparsity pattern."""
        self._ctx.pop("factorization", None)
        self._ctx.pop("pattern", None)


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
