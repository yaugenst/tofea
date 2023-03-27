import numpy as np
import pytest
from autograd.test_util import check_grads
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from tofea.primitives import solve_coo


@pytest.fixture
def rng():
    seed = 36523523
    return np.random.default_rng(seed)


@pytest.mark.parametrize("n", [10, 11])
def test_solve_coo(rng, n):
    m = rng.random((n, n))
    mx = np.sum(np.abs(m), axis=1)
    np.fill_diagonal(m, mx)
    m = coo_matrix(m)

    b = rng.random(n)

    x0 = spsolve(m.tocsr(), b)
    x1 = solve_coo(m.data, (m.row, m.col), b)

    assert_allclose(x0, x1)


@pytest.mark.parametrize("n", [10, 11])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_solve_coo_entries_grad(rng, n, mode):
    m = rng.random((n, n))
    mx = np.sum(np.abs(m), axis=1)
    np.fill_diagonal(m, mx)
    m = coo_matrix(m)

    b = rng.random(n)

    check_grads(lambda x: solve_coo(x, (m.row, m.col), b), modes=[mode], order=1)(
        m.data
    )


@pytest.mark.parametrize("n", [10, 11])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_solve_coo_b_grad(rng, n, mode):
    m = rng.random((n, n))
    mx = np.sum(np.abs(m), axis=1)
    np.fill_diagonal(m, mx)
    m = coo_matrix(m)

    b = rng.random(n)

    check_grads(lambda x: solve_coo(m.data, (m.row, m.col), x), modes=[mode], order=1)(
        b
    )
