from autograd.extend import defvjp, primitive
from scipy.sparse.linalg import coo_matrix, factorized

_ctx = {}


@primitive
def solve_coo(entries, indices, rhs):
    a = coo_matrix((entries, indices)).tocsc()
    _ctx["factorization"] = factorized(a)
    return _ctx["factorization"](rhs)


def solve_coo_entries_vjp(ans, entries, indices, b):
    def vjp(g):
        adjoint = _ctx["factorization"](g)  # TODO: generalize using U^T L^T!
        i, j = indices
        return -adjoint[i] * ans[j]

    return vjp


def solve_coo_b_vjp(ans, entries, indices, b):
    def vjp(g):
        return _ctx["factorization"](g)

    return vjp


defvjp(solve_coo, solve_coo_entries_vjp, None, solve_coo_b_vjp)
