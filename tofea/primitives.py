import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix

from tofea import DEFAULT_SOLVER
from tofea.solvers import get_solver

solver = get_solver(DEFAULT_SOLVER)


def _solve_coo(entries, indices, rhs):
    a = coo_matrix((entries, indices)).tocsc()
    solver.factor(a)
    return solver.solve(rhs)


@jax.custom_vjp
def solve_coo(entries, indices, rhs):
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(rhs.shape), dtype=entries.dtype
    )
    return jax.pure_callback(_solve_coo, result_shape_dtype, entries, indices, rhs)


def _solve_coo_fwd(entries, indices, rhs):
    x = solve_coo(entries, indices, rhs)
    return x, (x, entries, indices, rhs)


def _solve_coo_bwd(res, g):
    ans, entries, indices, rhs = res
    x = solve_coo(entries, indices, g)
    i, j = indices
    return -x[i] * ans[j], None, x


solve_coo.defvjp(_solve_coo_fwd, _solve_coo_bwd)
