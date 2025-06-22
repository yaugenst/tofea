import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from tofea.primitives import solve_coo
from tofea.solvers import get_solver


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)


@pytest.mark.parametrize("solver", ["SuperLU"])
@pytest.mark.parametrize("n", [10, 11])
def test_solve_coo(rng, solver, n):
    m = rng.random((n, n))
    m = coo_matrix(m @ m.T)

    b = rng.random(n)

    _solver = get_solver(solver)

    x0 = spsolve(m.tocsc(), b)
    x1 = solve_coo(m.data, (m.row, m.col), b, solver=_solver)

    assert_allclose(x0, x1)


def test_solver_instances_are_independent(rng):
    a = rng.random((5, 5))
    b = rng.random((5, 5))

    m1 = coo_matrix(a @ a.T)
    m2 = coo_matrix(b @ b.T)

    rhs1 = rng.random(5)
    rhs2 = rng.random(5)

    solver1 = get_solver("SuperLU")
    solver2 = get_solver("SuperLU")

    x1 = solve_coo(m1.data, (m1.row, m1.col), rhs1, solver=solver1)
    _ = solve_coo(m2.data, (m2.row, m2.col), rhs2, solver=solver2)
    y1 = solve_coo(m1.data, (m1.row, m1.col), rhs1, solver=solver1)

    assert_allclose(x1, y1)
