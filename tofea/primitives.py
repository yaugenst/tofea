from autograd.extend import defjvp, defvjp, primitive
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

_ctx = {}


@primitive
def solve_coo(entries, indices, rhs):
    a = coo_matrix((entries, indices)).tocsc()
    _ctx["factorization"] = splu(
        a,
        diag_pivot_thresh=0.1,
        permc_spec="MMD_AT_PLUS_A",
        options=dict(SymmetricMode=True),
    )
    return _ctx["factorization"].solve(rhs)


def solve_coo_entries_jvp(g, x, entries, indices, b):
    a = coo_matrix((g, indices)).tocsc()
    return _ctx["factorization"].solve(-(a @ x))


def solve_coo_b_jvp(g, x, entries, indices, b):
    return _ctx["factorization"].solve(g)


defjvp(solve_coo, solve_coo_entries_jvp, None, solve_coo_b_jvp)


def solve_coo_entries_vjp(ans, entries, indices, b):
    def vjp(g):
        x = _ctx["factorization"].solve(g, trans="T")
        i, j = indices
        return -x[i] * ans[j]

    return vjp


def solve_coo_b_vjp(ans, entries, indices, b):
    def vjp(g):
        return _ctx["factorization"].solve(g, trans="T")

    return vjp


defvjp(solve_coo, solve_coo_entries_vjp, None, solve_coo_b_vjp)
