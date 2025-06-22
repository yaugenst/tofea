import numpy as np

from tofea.boundary_conditions import BoundaryConditions
from tofea.fea2d import FEA2D_K, FEA2D_T


def test_integration_with_fea2d():
    """BoundaryConditions interact correctly with FEA2D models."""

    # Heat conduction
    bc = BoundaryConditions((1, 1))
    bc.fix_edge("left")
    bc.apply_point_load(1, 1, 1.0)

    fem = FEA2D_T(bc.fixed)
    x = np.ones(fem.shape)
    t = fem.temperature(x, bc.load).reshape((2, 2))

    assert np.allclose(t[0], 0.0)  # fixed edge
    assert t[1, 1] > 0

    # Elasticity
    bc2 = BoundaryConditions((1, 1), dof_dim=2)
    bc2.fix_edge("bottom")
    bc2.apply_point_load(1, 1, (1.0, 0.0))

    fem2 = FEA2D_K(bc2.fixed)
    x2 = np.ones(fem2.shape)
    u = fem2.displacement(x2, bc2.load).reshape((2, 2, 2))

    assert np.allclose(u[:, 0, :], 0.0)  # fixed edge
    assert u[1, 1, 0] > 0
