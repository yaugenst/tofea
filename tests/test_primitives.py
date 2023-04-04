import numpy as np
import pytest
from autograd.test_util import check_grads
from scipy.sparse import coo_matrix

from tofea.primitives import solve_coo


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)


@pytest.mark.parametrize("n", [10, 11])
@pytest.mark.parametrize(
    "solver",
    [
        "scipy",
        "pardiso",
        pytest.param(
            "cholesky",
            marks=pytest.mark.xfail(
                reason="Cholesky entries grads are wrong currently."
            ),
        ),
        "umfpack",
        "gpu",
    ],
)
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_solve_coo_entries_grad(rng, n, solver, mode):
    m = rng.random((n, n))
    m = coo_matrix(m @ m.T)

    b = rng.random(n)

    check_grads(
        lambda x: solve_coo(x, (m.row, m.col), b, solver), modes=[mode], order=1
    )(m.data)


@pytest.mark.parametrize("n", [10, 11])
@pytest.mark.parametrize("solver", ["scipy", "pardiso", "cholesky", "umfpack", "gpu"])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_solve_coo_b_grad(rng, n, solver, mode):
    m = rng.random((n, n))
    m = coo_matrix(m @ m.T)

    b = rng.random(n)

    check_grads(
        lambda x: solve_coo(m.data, (m.row, m.col), x, solver), modes=[mode], order=1
    )(b)
