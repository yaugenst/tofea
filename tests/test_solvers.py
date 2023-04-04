import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from tofea.primitives import solve_coo


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)


@pytest.mark.parametrize("solver", ["scipy", "pardiso", "cholesky", "umfpack", "gpu"])
@pytest.mark.parametrize("n", [10, 11])
def test_solve_coo(rng, solver, n):
    m = rng.random((n, n))
    m = coo_matrix(m @ m.T)

    b = rng.random(n)

    x0 = spsolve(m.tocsc(), b)
    x1 = solve_coo(m.data, (m.row, m.col), b, solver=solver)

    assert_allclose(x0, x1)
