"""Autograd primitives used by the finite element routines."""

from collections.abc import Callable

import autograd.numpy as anp
import numpy as np
from autograd.extend import defjvp, defvjp, primitive
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix

from tofea.solvers import Solver


def _pattern_signature(m: csc_matrix) -> tuple[NDArray[np.int_], NDArray[np.int_], tuple[int, int]]:
    """Return a signature describing the sparsity pattern of ``m``."""

    return m.indices.copy(), m.indptr.copy(), m.shape


def _same_pattern(
    m: csc_matrix, sig: tuple[NDArray[np.int_], NDArray[np.int_], tuple[int, int]]
) -> bool:
    """Return ``True`` if ``m`` matches the sparsity pattern ``sig``."""

    indices, indptr, shape = sig
    return (
        m.shape == shape and np.array_equal(m.indices, indices) and np.array_equal(m.indptr, indptr)
    )


@primitive
def solve_coo(
    entries: NDArray,
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],
    rhs: NDArray,
    solver: Solver,
) -> NDArray:
    """Solve a sparse linear system in COO format.

    Parameters
    ----------
    entries : array-like
        Non-zero matrix entries.
    indices : tuple[array-like, array-like]
        Row and column indices for ``entries``.
    rhs : array-like
        Right-hand side vector.
    solver : tofea.solvers.Solver
        Factorization backend used to solve the system.

    Returns
    -------
    numpy.ndarray
        Solution vector.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> from tofea.primitives import solve_coo
    >>> from tofea.solvers import get_solver
    >>> m = coo_matrix([[4, 1], [1, 3]])
    >>> b = np.array([1, 2])
    >>> solve_coo(m.data, (m.row, m.col), b, get_solver("SuperLU"))
    array([0.09090909, 0.63636364])
    """

    a = coo_matrix((entries, indices)).tocsc()
    ctx = getattr(solver, "_ctx", {})
    pattern = ctx.get("pattern")
    if pattern is not None and _same_pattern(a, pattern):
        solver.refactor_numerical(a)
    else:
        solver.clear()
        solver.factor_full(a)
        ctx["pattern"] = _pattern_signature(a)
    return solver.solve(rhs)


def solve_coo_entries_jvp(
    g: NDArray,
    x: NDArray,
    entries: NDArray,  # noqa: ARG001
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],
    rhs: NDArray,  # noqa: ARG001
    solver: Solver,
) -> NDArray:
    """Forward-mode derivative of :func:`solve_coo` with respect to ``entries``."""

    a = coo_matrix((g, indices)).tocsc()
    return solver.solve(-(a @ x))


def solve_coo_b_jvp(
    g: NDArray,
    x: NDArray,  # noqa: ARG001
    entries: NDArray,  # noqa: ARG001
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],  # noqa: ARG001
    rhs: NDArray,  # noqa: ARG001
    solver: Solver,
) -> NDArray:
    """Forward-mode derivative of :func:`solve_coo` with respect to ``rhs``."""

    return solver.solve(g)


defjvp(solve_coo, solve_coo_entries_jvp, None, solve_coo_b_jvp)


def solve_coo_entries_vjp(
    ans: NDArray,
    entries: NDArray,  # noqa: ARG001
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],
    rhs: NDArray,  # noqa: ARG001
    solver: Solver,
) -> Callable[[NDArray], NDArray]:
    """Reverse-mode derivative of :func:`solve_coo` for ``entries``."""

    def vjp(g: NDArray) -> NDArray:
        x = solver.solve(g, transpose=True)
        i, j = indices
        return -x[i] * ans[j]

    return vjp


def solve_coo_b_vjp(
    ans: NDArray,  # noqa: ARG001
    entries: NDArray,  # noqa: ARG001
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],  # noqa: ARG001
    rhs: NDArray,  # noqa: ARG001
    solver: Solver,
) -> Callable[[NDArray], NDArray]:
    """Reverse-mode derivative of :func:`solve_coo` for ``rhs``."""

    def vjp(g: NDArray) -> NDArray:
        return solver.solve(g, transpose=True)

    return vjp


defvjp(solve_coo, solve_coo_entries_vjp, None, solve_coo_b_vjp)


@primitive
def solve_and_compute_self_adjoint_objective(
    entries: NDArray,
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],
    rhs: NDArray,
    solver: Solver,
) -> tuple[NDArray, NDArray]:
    """Solve ``K u = b`` and return the self-adjoint objective and solution."""

    u_nz = solve_coo(entries, indices, rhs, solver)
    objective = anp.dot(u_nz, rhs)
    return objective, u_nz


def solve_and_compute_self_adjoint_objective_vjp(
    ans: tuple[NDArray, NDArray],
    entries: NDArray,  # noqa: ARG001
    indices: tuple[NDArray[np.int_], NDArray[np.int_]],
    rhs: NDArray,  # noqa: ARG001
    solver: Solver,  # noqa: ARG001
) -> Callable[[NDArray | tuple[NDArray, NDArray]], NDArray]:
    """Reverse-mode derivative for ``solve_and_compute_self_adjoint_objective``."""

    _, u_nz = ans

    def vjp(g: NDArray | tuple[NDArray, NDArray]) -> NDArray:
        if not isinstance(g, np.ndarray):
            g = g[0]
        i, j = indices
        return -g * u_nz[i] * u_nz[j]

    return vjp


defvjp(
    solve_and_compute_self_adjoint_objective,
    solve_and_compute_self_adjoint_objective_vjp,
    None,
    None,
    None,
)
