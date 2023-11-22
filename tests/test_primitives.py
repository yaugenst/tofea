import pytest
from jax.test_util import check_grads
from scipy.sparse import coo_matrix

from tofea.primitives import solve_coo


@pytest.mark.parametrize("n", [10, 11])
def test_solve_coo_entries_grad(rng, n):
    m = rng.random((n, n))
    m = coo_matrix(m @ m.T)

    b = rng.random(n)

    check_grads(
        lambda x: solve_coo(x, (m.row, m.col), b),
        (m.data,),
        order=1,
        modes=["rev"],
        atol=10000,
        rtol=1e-3,
    )


@pytest.mark.parametrize("n", [10, 11])
def test_solve_coo_b_grad(rng, n):
    m = rng.random((n, n))
    m = coo_matrix(m @ m.T)

    b = rng.random(n)

    check_grads(
        lambda x: solve_coo(m.data, (m.row, m.col), x),
        (b,),
        order=1,
        modes=["rev"],
        atol=10000,
        rtol=1e-3,
    )
