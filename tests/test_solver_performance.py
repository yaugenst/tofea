import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix

from tofea.fea2d import FEA2D_T
from tofea.primitives import solve_coo
from tofea.solvers import Solver, SuperLU, get_solver


def rng():
    seed = 654321
    return np.random.default_rng(seed)


def test_superlu_api():
    solver: Solver = SuperLU()
    m = coo_matrix(np.eye(2)).tocsc()
    solver.factor_full(m)
    solver.refactor_numerical(m)
    out = solver.solve(np.array([1.0, 0.0]))
    assert out.shape == (2,)
    solver.clear()


def test_cached_vs_uncached_solution():
    r = rng()
    m = r.random((5, 5))
    m = coo_matrix(m @ m.T)
    b = r.random(5)

    s1 = get_solver("SuperLU")
    u0 = solve_coo(m.data, (m.row, m.col), b, s1)
    s1.clear()
    u1 = solve_coo(m.data, (m.row, m.col), b, s1)

    s2 = get_solver("SuperLU")
    v0 = solve_coo(m.data, (m.row, m.col), b, s2)
    v1 = solve_coo(m.data, (m.row, m.col), b, s2)

    assert_allclose(u0, v0)
    assert_allclose(u1, v1)


def test_pattern_change_triggers_full_factorization():
    """Ensure cached factorization is discarded when the sparsity pattern changes."""

    s = get_solver("SuperLU")

    m1 = coo_matrix(np.eye(3))
    b = np.array([1.0, 2.0, 3.0])
    solve_coo(m1.data, (m1.row, m1.col), b, s)
    pattern1 = s._ctx.get("pattern")

    m2 = coo_matrix([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    solve_coo(m2.data, (m2.row, m2.col), b, s)
    pattern2 = s._ctx.get("pattern")

    assert not _arrays_identical(pattern1, pattern2)


def _arrays_identical(p1, p2):
    if p1 is None or p2 is None:
        return False
    i1, ptr1, shape1 = p1
    i2, ptr2, shape2 = p2
    return shape1 == shape2 and np.array_equal(i1, i2) and np.array_equal(ptr1, ptr2)


def test_cached_gradient_matches_standard():
    fixed = np.zeros((3, 3), dtype=bool)
    fixed[0, 0] = True
    fem1 = FEA2D_T(fixed)
    fem2 = FEA2D_T(fixed)
    load = np.zeros_like(fixed, dtype=float)
    load[-1, -1] = 1.0
    x = np.full(fem1.shape, 0.5)

    def obj_standard(x_):
        t = fem1.temperature(x_, load)
        return anp.mean(t)

    def obj_cached(x_):
        t = fem2.temperature(x_, load)
        return anp.mean(t)

    val_std, grad_std = value_and_grad(obj_standard)(x)
    val_cache, grad_cache = value_and_grad(obj_cached)(x)

    assert_allclose(val_std, val_cache)
    assert_allclose(grad_std, grad_cache)
