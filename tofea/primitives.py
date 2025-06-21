"""Autograd primitives used by the finite element routines."""

from autograd.extend import defjvp, defvjp, primitive
from scipy.sparse import coo_matrix


@primitive
def solve_coo(entries, indices, rhs, solver):
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
    solver.clear()
    solver.factor(a)
    return solver.solve(rhs)


def solve_coo_entries_jvp(g, x, entries, indices, rhs, solver):  # noqa: ARG001
    """Forward-mode derivative of :func:`solve_coo` with respect to ``entries``."""

    a = coo_matrix((g, indices)).tocsc()
    return solver.solve(-(a @ x))


def solve_coo_b_jvp(g, x, entries, indices, rhs, solver):  # noqa: ARG001
    """Forward-mode derivative of :func:`solve_coo` with respect to ``rhs``."""

    return solver.solve(g)


defjvp(solve_coo, solve_coo_entries_jvp, None, solve_coo_b_jvp)


def solve_coo_entries_vjp(ans, entries, indices, rhs, solver):  # noqa: ARG001
    """Reverse-mode derivative of :func:`solve_coo` for ``entries``."""

    def vjp(g):
        x = solver.solve(g, transpose=True)
        i, j = indices
        return -x[i] * ans[j]

    return vjp


def solve_coo_b_vjp(ans, entries, indices, rhs, solver):  # noqa: ARG001
    """Reverse-mode derivative of :func:`solve_coo` for ``rhs``."""

    def vjp(g):
        return solver.solve(g, transpose=True)

    return vjp


defvjp(solve_coo, solve_coo_entries_vjp, None, solve_coo_b_vjp)
