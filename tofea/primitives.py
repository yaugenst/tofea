from autograd.extend import defjvp, defvjp, primitive
from scipy.sparse import coo_matrix

from tofea.solvers import scipy_solver as solver


@primitive
def solve_coo(entries, indices, rhs):
    a = coo_matrix((entries, indices)).tocsc()
    solver.clear()
    solver.factor(a)
    return solver.solve(rhs)


def solve_coo_entries_jvp(g, x, entries, indices, b):
    a = coo_matrix((g, indices)).tocsc()
    return solver.solve(-(a @ x))


def solve_coo_b_jvp(g, x, entries, indices, b):
    return solver.solve(g)


defjvp(solve_coo, solve_coo_entries_jvp, None, solve_coo_b_jvp)


def solve_coo_entries_vjp(ans, entries, indices, b):
    def vjp(g):
        x = solver.solve(g, transpose=True)
        i, j = indices
        return -x[i] * ans[j]

    return vjp


def solve_coo_b_vjp(ans, entries, indices, b):
    def vjp(g):
        return solver.solve(g, transpose=True)

    return vjp


defvjp(solve_coo, solve_coo_entries_vjp, None, solve_coo_b_vjp)
