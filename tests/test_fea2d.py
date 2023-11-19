import numpy as np
import pytest

from tofea.fea2d import FEA2D_K, FEA2D_T


class TestFEA2DK:
    @pytest.fixture()
    def fea2d_k_instance(self):
        fixed = np.zeros((5, 5, 2), dtype=bool)
        return FEA2D_K(fixed)

    def test_shape(self, fea2d_k_instance):
        assert fea2d_k_instance.shape == (4, 4)

    def test_dofs(self, fea2d_k_instance):
        dofs = fea2d_k_instance.dofs
        assert isinstance(dofs, np.ndarray)
        assert dofs.shape == (5, 5, 2)
        assert np.all(dofs == np.arange(50).reshape(5, 5, 2))

    def test_fixdofs(self, fea2d_k_instance):
        fixdofs = fea2d_k_instance.fixdofs
        assert isinstance(fixdofs, np.ndarray)
        assert fixdofs.size == 0

    def test_freedofs(self, fea2d_k_instance):
        freedofs = fea2d_k_instance.freedofs
        assert isinstance(freedofs, np.ndarray)
        assert np.all(freedofs == np.arange(50))


class TestFEA2DT:
    @pytest.fixture()
    def fea2d_t_instance(self):
        fixed = np.zeros((5, 5), dtype=bool)
        return FEA2D_T(fixed)

    def test_shape(self, fea2d_t_instance):
        assert fea2d_t_instance.shape == (4, 4)

    def test_dofs(self, fea2d_t_instance):
        dofs = fea2d_t_instance.dofs
        assert isinstance(dofs, np.ndarray)
        assert dofs.shape == (5, 5)
        assert np.all(dofs == np.arange(25).reshape(5, 5))

    def test_fixdofs(self, fea2d_t_instance):
        fixdofs = fea2d_t_instance.fixdofs
        assert isinstance(fixdofs, np.ndarray)
        assert fixdofs.size == 0

    def test_freedofs(self, fea2d_t_instance):
        freedofs = fea2d_t_instance.freedofs
        assert isinstance(freedofs, np.ndarray)
        assert np.all(freedofs == np.arange(25))
