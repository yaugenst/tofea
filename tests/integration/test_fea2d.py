import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from tofea.fea2d import FEA2D_K, FEA2D_T


def test_fea2d_k_solution_matches_spsolve():
    fixed = np.zeros((2, 2, 2), dtype=bool)
    fixed[0] = True
    fem = FEA2D_K(fixed)
    x = np.array([[1.0]])
    load = np.zeros_like(fixed, dtype=float)
    load[1, 1, 1] = 1.0

    disp = fem.displacement(x, load)

    data, indices = fem.global_mat(x)
    k = coo_matrix(
        (data, (indices[0], indices[1])), shape=(len(fem.freedofs), len(fem.freedofs))
    ).tocsc()
    u_free = spsolve(k, load.ravel()[fem.freedofs])
    expected = np.concatenate([u_free, np.zeros(len(fem.fixdofs))])[fem.index_map]

    assert_allclose(disp, expected)


def test_fea2d_t_solution_matches_spsolve():
    fixed = np.zeros((2, 2), dtype=bool)
    fixed[0, 0] = True
    fem = FEA2D_T(fixed)
    x = np.array([[1.0]])
    load = np.zeros_like(fixed, dtype=float)
    load[1, 1] = 1.0

    temp = fem.temperature(x, load)

    data, indices = fem.global_mat(x)
    k = coo_matrix(
        (data, (indices[0], indices[1])), shape=(len(fem.freedofs), len(fem.freedofs))
    ).tocsc()
    t_free = spsolve(k, load.ravel()[fem.freedofs])
    expected = np.concatenate([t_free, np.zeros(len(fem.fixdofs))])[fem.index_map]

    assert_allclose(temp, expected)
