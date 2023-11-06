import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix


class AbstractSolver(ABC):
    _ctx: dict[int, Any] = {}

    @abstractmethod
    def factor(self, m: csc_matrix | csr_matrix) -> None:
        ...

    @abstractmethod
    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class SciPySolver(AbstractSolver):
    def __init__(self, **options):
        from scipy.sparse.linalg import splu

        self._ctx["splu"] = partial(splu, **options)

    def factor(self, m: csc_matrix) -> None:
        self._ctx["factorization"] = self._ctx["splu"](m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        return self._ctx["factorization"].solve(rhs, trans="T" if transpose else "N")

    def clear(self) -> None:
        self._ctx.pop("factorization", None)


scipy_solver = SciPySolver(
    diag_pivot_thresh=0.1,
    permc_spec="MMD_AT_PLUS_A",
    options={"SymmetricMode": True},
)


def get_solver(solver):
    return scipy_solver
