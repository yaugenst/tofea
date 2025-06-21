import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad
from numpy.testing import assert_allclose

from tofea.fea2d import FEA2D_K, FEA2D_T


def test_self_adjoint_vs_generic_compliance():
    fixed = np.zeros((3, 3, 2), dtype=bool)
    fixed[0] = 1
    fem = FEA2D_K(fixed)
    x = np.full(fem.shape, 0.5)
    load = np.zeros_like(fixed, dtype=float)
    load[-1, -1, 1] = 1.0

    def generic_obj(x_):
        return anp.dot(fem.displacement(x_, load).ravel(), load.ravel())

    val_gen, grad_gen = value_and_grad(generic_obj)(x)

    val_sa, grad_sa = value_and_grad(lambda x_: fem.compliance(x_, load))(x)

    assert_allclose(val_gen, val_sa)
    assert_allclose(grad_gen, grad_sa)


def test_self_adjoint_vs_generic_thermal_compliance():
    fixed = np.zeros((3, 3), dtype=bool)
    fixed[0, 0] = 1
    fem = FEA2D_T(fixed)
    x = np.full(fem.shape, 0.5)
    load = np.zeros_like(fixed, dtype=float)
    load[-1, -1] = 1.0

    def generic_obj(x_):
        return anp.dot(fem.temperature(x_, load).ravel(), load.ravel())

    val_gen, grad_gen = value_and_grad(generic_obj)(x)

    val_sa, grad_sa = value_and_grad(lambda x_: fem.thermal_compliance(x_, load))(x)

    assert_allclose(val_gen, val_sa)
    assert_allclose(grad_gen, grad_sa)
