from autograd.extend import defjvp, defvjp, primitive
from scipy.sparse import coo_matrix

from tofea.solvers import get_solver

_default_solver = "scipy"


@primitive
def solve_coo(entries, indices, rhs, solver=_default_solver):
    a = coo_matrix((entries, indices)).tocsc()
    _solver = get_solver(solver)
    _solver.clear()
    _solver.factor(a)
    return _solver.solve(rhs)


def solve_coo_entries_jvp(g, x, entries, indices, b, solver=_default_solver):
    a = coo_matrix((g, indices)).tocsc()
    _solver = get_solver(solver)
    return _solver.solve(-(a @ x))


def solve_coo_b_jvp(g, x, entries, indices, b, solver=_default_solver):
    _solver = get_solver(solver)
    return _solver.solve(g)


defjvp(solve_coo, solve_coo_entries_jvp, None, solve_coo_b_jvp)


def solve_coo_entries_vjp(ans, entries, indices, b, solver=_default_solver):
    def vjp(g):
        _solver = get_solver(solver)
        x = _solver.solve(g, transpose=True)
        i, j = indices
        return -x[i] * ans[j]

    return vjp


def solve_coo_b_vjp(ans, entries, indices, b, solver=_default_solver):
    def vjp(g):
        _solver = get_solver(solver)
        return _solver.solve(g, transpose=True)

    return vjp


defvjp(solve_coo, solve_coo_entries_vjp, None, solve_coo_b_vjp)
