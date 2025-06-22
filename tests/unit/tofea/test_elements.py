import numpy as np
import pytest

from tofea.elements import Q4Element, Q4Element_K, Q4Element_T


class TestQ4ElementK:
    @pytest.fixture
    def q4element_k_instance(self):
        return Q4Element_K(e=1.0, nu=1 / 3, dx=0.5, dy=0.5)

    def test_element(self, q4element_k_instance):
        element = q4element_k_instance.element
        assert isinstance(element, np.ndarray)
        assert element.shape == (8, 8)


class TestQ4ElementT:
    @pytest.fixture
    def q4element_t_instance(self):
        return Q4Element_T(k=1.0, dx=0.5, dy=0.5)

    def test_element(self, q4element_t_instance):
        element = q4element_t_instance.element
        assert isinstance(element, np.ndarray)
        assert element.shape == (4, 4)


def test_q4element_utilities():
    elem = Q4Element()
    points = elem.gauss_points()
    assert len(points) == 4
    dxi, deta = elem.grad_shape_funcs(0.0, 0.0)
    np.testing.assert_allclose(dxi, [-0.25, 0.25, 0.25, -0.25])
    np.testing.assert_allclose(deta, [-0.25, -0.25, 0.25, 0.25])
