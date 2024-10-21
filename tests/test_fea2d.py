import numpy as np
import pytest
from autograd.test_util import check_grads

from tofea.fea2d import FEA2D_K, FEA2D_T


@pytest.fixture
def rng():
    seed = 36523523
    return np.random.default_rng(seed)


class TestFEA2DK:
    @pytest.fixture
    def fea2d_k_instance(self):
        fixed = np.zeros((5, 5, 2), dtype=bool)
        fixed[0] = 1
        return FEA2D_K(fixed)

    @pytest.fixture
    def x_and_b(self, fea2d_k_instance, rng):
        x = rng.random(fea2d_k_instance.shape)
        b = rng.random(fea2d_k_instance.fixed.shape)
        return x, b

    def test_shape(self, fea2d_k_instance):
        assert fea2d_k_instance.shape == (4, 4)

    def test_dofs(self, fea2d_k_instance):
        dofs = fea2d_k_instance.dofs
        assert isinstance(dofs, np.ndarray)
        assert dofs.shape == (50,)
        assert np.all(dofs == np.arange(50))

    def test_fixdofs(self, fea2d_k_instance):
        fixdofs = fea2d_k_instance.fixdofs
        assert isinstance(fixdofs, np.ndarray)
        assert fixdofs.size == fea2d_k_instance.fixed[0].size

    def test_freedofs(self, fea2d_k_instance):
        freedofs = fea2d_k_instance.freedofs
        assert isinstance(freedofs, np.ndarray)
        assert freedofs.size == 50 - fea2d_k_instance.fixdofs.size

    def test_displacement_grads(self, fea2d_k_instance, x_and_b):
        x, b = x_and_b
        check_grads(
            lambda x_: fea2d_k_instance.displacement(x_, b),
            modes=["fwd", "rev"],
            order=1,
        )(x)
        check_grads(
            lambda b_: fea2d_k_instance.displacement(x, b_),
            modes=["fwd", "rev"],
            order=1,
        )(b)

    def test_compliance_grads(self, fea2d_k_instance, x_and_b, rng):
        x, _ = x_and_b
        d = rng.random(fea2d_k_instance.dofs.shape)
        check_grads(
            lambda x_: fea2d_k_instance.compliance(x_, d), modes=["fwd", "rev"], order=1
        )(x)
        check_grads(
            lambda d_: fea2d_k_instance.compliance(x, d_), modes=["fwd", "rev"], order=1
        )(d)


class TestFEA2DT:
    @pytest.fixture
    def fea2d_t_instance(self):
        fixed = np.zeros((5, 5), dtype=bool)
        fixed[0, 0] = 1
        return FEA2D_T(fixed)

    @pytest.fixture
    def x_and_b(self, fea2d_t_instance, rng):
        x = rng.random(fea2d_t_instance.shape)
        b = rng.random(fea2d_t_instance.fixed.shape)
        return x, b

    def test_shape(self, fea2d_t_instance):
        assert fea2d_t_instance.shape == (4, 4)

    def test_dofs(self, fea2d_t_instance):
        dofs = fea2d_t_instance.dofs
        assert isinstance(dofs, np.ndarray)
        assert dofs.shape == (25,)
        assert np.all(dofs == np.arange(25))

    def test_fixdofs(self, fea2d_t_instance):
        fixdofs = fea2d_t_instance.fixdofs
        assert isinstance(fixdofs, np.ndarray)
        assert fixdofs.size == 1

    def test_freedofs(self, fea2d_t_instance):
        freedofs = fea2d_t_instance.freedofs
        assert isinstance(freedofs, np.ndarray)
        assert freedofs.size == 25 - fea2d_t_instance.fixdofs.size

    def test_temperature_grads(self, fea2d_t_instance, x_and_b):
        x, b = x_and_b
        check_grads(
            lambda x_: fea2d_t_instance.temperature(x_, b),
            modes=["fwd", "rev"],
            order=1,
        )(x)
        check_grads(
            lambda b_: fea2d_t_instance.temperature(x, b_),
            modes=["fwd", "rev"],
            order=1,
        )(b)
