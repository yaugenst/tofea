import numpy as np
import pytest

from tofea.boundary_conditions import BoundaryConditions
from tofea.fea2d import FEA2D_K, FEA2D_T


def test_fix_edge_1d():
    bc = BoundaryConditions((10, 10))
    bc.fix_edge("left")
    assert bc.fixed.shape == (11, 11)
    assert np.all(bc.fixed[0])
    assert not np.any(bc.fixed[1:])

    bc = BoundaryConditions((10, 10))
    bc.fix_edge("top")
    assert np.all(bc.fixed[:, -1])


def test_fix_edge_2d():
    bc = BoundaryConditions((5, 5), dof_dim=2)
    bc.fix_edge("bottom")
    assert bc.fixed.shape == (6, 6, 2)
    assert np.all(bc.fixed[:, 0])
    assert not np.any(bc.fixed[:, 1:])


def test_invalid_edge():
    bc = BoundaryConditions((2, 2))
    with pytest.raises(ValueError, match="edge"):
        bc.fix_edge("middle")


def test_apply_point_load():
    bc = BoundaryConditions((2, 2))
    bc.apply_point_load(1, 1, 5.0)
    assert bc.load[1, 1] == 5.0

    bc2 = BoundaryConditions((2, 2), dof_dim=2)
    bc2.apply_point_load(0, 0, (1.0, 2.0))
    np.testing.assert_allclose(bc2.load[0, 0], [1.0, 2.0])


def test_apply_uniform_load_on_edge():
    bc = BoundaryConditions((3, 3))
    bc.apply_uniform_load_on_edge("right", 2.0)
    assert np.all(bc.load[-1] == 2.0)

    bc2 = BoundaryConditions((3, 3), dof_dim=2)
    bc2.apply_uniform_load_on_edge("left", (1.0, 0.0))
    assert np.allclose(bc2.load[0], [1.0, 0.0])


def test_integration_with_fea2d():
    bc = BoundaryConditions((1, 1))
    fem = FEA2D_T(bc.fixed)
    x = np.ones(fem.shape)
    fem.temperature(x, bc.load)

    bc2 = BoundaryConditions((1, 1), dof_dim=2)
    fem2 = FEA2D_K(bc2.fixed)
    x2 = np.ones(fem2.shape)
    fem2.displacement(x2, bc2.load)
