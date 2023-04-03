import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix


class AbstractSolver(ABC):
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
    _ctx: dict[Any] = {}

    def __init__(self, **options):
        from scipy.sparse.linalg import splu

        self._ctx["splu"] = partial(splu, **options)

    def factor(self, m: csc_matrix) -> None:
        self._ctx["factorization"] = self._ctx["splu"](m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        return self._ctx["factorization"].solve(rhs, trans="T" if transpose else "N")

    def clear(self) -> None:
        self._ctx.pop("factorization", None)


class PardisoSolver(AbstractSolver):
    _ctx: dict[Any] = {}

    def __init__(self, **options):
        from pyMKL import pardisoSolver

        self._ctx["pardiso"] = partial(pardisoSolver, **options)

    def factor(self, m: csr_matrix | csc_matrix) -> None:
        self._ctx["factorization"] = self._ctx["pardiso"](m)
        self._ctx["factorization"].factor()

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        """Warning! Transpose does nothing because pyMKL does not expose iparm
        This is not a problem in the case of tofea since matrices are guaranteed to be
        symmetric, but gradients will be wrong if that is not the case.
        Probably will need to implement a wrapper for MKL at some point.
        """
        return self._ctx["factorization"].solve(rhs)

    def clear(self) -> None:
        if "factorization" in self._ctx:
            self._ctx["factorization"].clear()
        self._ctx.pop("factorization", None)


class CholeskySolver(AbstractSolver):
    _ctx: dict[Any] = {}

    def __init__(self, **options):
        from sksparse.cholmod import cholesky

        self._ctx["cholesky"] = partial(cholesky, **options)

    def factor(self, m: csc_matrix) -> None:
        if "factorization" in self._ctx:
            self._ctx["factorization"].cholesky_inplace(m)
        else:
            self._ctx["factorization"] = self._ctx["cholesky"](m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        """Transpose does nothing because Cholesky decomposition only works for real
        symmetric matrices...
        """
        return self._ctx["factorization"].solve_A(rhs)

    def clear(self) -> None:
        self._ctx.pop("factorization", None)


class UmfpackSolver(AbstractSolver):
    _ctx: dict[Any] = {}

    def __init__(self, **options):
        from scikits.umfpack import UMFPACK_A, UMFPACK_At, UmfpackContext

        self._ctx["modes"] = {"N": UMFPACK_A, "T": UMFPACK_At}
        self._ctx["umfpack"] = UmfpackContext(**options)

    def factor(self, m: csc_matrix) -> None:
        self._ctx["umfpack"].numeric(m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        mode = self._ctx["modes"]["T"] if transpose else self._ctx["modes"]["N"]
        return self._ctx["umfpack"].solve(
            mode, self._ctx["umfpack"].mtx, rhs, autoTranspose=True
        )

    def clear(self) -> None:
        self._ctx["umfpack"].free_numeric()


scipy_solver = SciPySolver(
    diag_pivot_thresh=0.1,
    permc_spec="MMD_AT_PLUS_A",
    options=dict(SymmetricMode=True),
)

try:
    pardiso_solver = PardisoSolver(mtype=11)
except ImportError:
    warnings.warn("pyMKL not found, pardiso_solver unavailable")

try:
    cholesky_solver = CholeskySolver()
except ImportError:
    warnings.warn("scikit-sparse not found, cholesky_solver unavailable")

try:
    umfpack_solver = UmfpackSolver()
except ImportError:
    warnings.warn("scikit-umfpack not found, umfpack_solver unavailable")


def get_solver(solver):
    match solver:
        case "scipy":
            return scipy_solver
        case "pardiso":
            return pardiso_solver
        case "cholesky":
            return cholesky_solver
        case "umfpack":
            return umfpack_solver
        case _:
            raise ValueError(f"Invalid solver: {solver}")
