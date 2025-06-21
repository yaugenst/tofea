"""Finite element primitives used by :mod:`tofea`."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class Element:
    """Base dataclass for finite elements."""

    dx: float = 0.5
    dy: float = 0.5
    dz: float = 0.5
    eps: float = 1e-6
    dtype: type = np.float64


class Q4Element(Element):
    """Four-node quadrilateral element.

    This class provides utilities shared by all quadrilateral elements
    implemented in :mod:`tofea`, namely Gauss quadrature points and the
    gradients of the bilinear shape functions.  Having these helpers in a
    common base class avoids code duplication in the concrete element
    implementations and gives this abstraction a clear purpose.
    """

    @staticmethod
    def gauss_points() -> list[tuple[float, float]]:
        """Return the points for 2x2 Gauss quadrature."""

        gp = 1 / np.sqrt(3)
        return [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    def grad_shape_funcs(self, xi: float, eta: float) -> tuple[NDArray, NDArray]:
        """Return shape function gradients for local coordinates."""

        dN_dxi: NDArray = np.array(
            [
                -0.25 * (1 - eta),
                0.25 * (1 - eta),
                0.25 * (1 + eta),
                -0.25 * (1 + eta),
            ],
            dtype=self.dtype,
        )
        dN_deta: NDArray = np.array(
            [
                -0.25 * (1 - xi),
                -0.25 * (1 + xi),
                0.25 * (1 + xi),
                0.25 * (1 - xi),
            ],
            dtype=self.dtype,
        )
        return dN_dxi, dN_deta


@dataclass(frozen=True, slots=True)
class Q4Element_K(Q4Element):
    """Plane stress elasticity element.

    Parameters
    ----------
    e : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    """

    e: float = 1.0
    nu: float = 1 / 3

    @cached_property
    def element(self) -> NDArray:
        """Return the 8x8 stiffness matrix for the element.

        The matrix is assembled using 2x2 Gauss quadrature instead of the
        symbolic integration previously used.  This avoids the costly SymPy
        computations and speeds up object construction significantly while
        maintaining numerical precision.

        Returns
        -------
        numpy.ndarray
            Stiffness matrix of shape ``(8, 8)``.

        Examples
        --------
        >>> from tofea.elements import Q4Element_K
        >>> Q4Element_K().element.shape
        (8, 8)
        """

        C: NDArray = (self.e / (1 - self.nu**2)) * np.array(
            [[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]],
            dtype=self.dtype,
        )

        K: NDArray = np.zeros((8, 8), dtype=self.dtype)

        for xi, eta in self.gauss_points():
            dN_dxi, dN_deta = self.grad_shape_funcs(xi, eta)

            B: NDArray = np.zeros((3, 8), dtype=self.dtype)
            for i in range(4):
                B[0, 2 * i] = dN_dxi[i] / self.dx
                B[1, 2 * i + 1] = dN_deta[i] / self.dy
                B[2, 2 * i] = dN_deta[i] / self.dy
                B[2, 2 * i + 1] = dN_dxi[i] / self.dx

            K += (B.T @ C @ B) * self.dx * self.dy

        K[np.abs(K) < self.eps] = 0
        return K


@dataclass(frozen=True, slots=True)
class Q4Element_T(Q4Element):
    """Heat conductivity element.

    Parameters
    ----------
    k : float
        Thermal conductivity of the material.
    """

    k: float = 1.0

    @cached_property
    def element(self) -> NDArray:
        """Return the 4x4 conductivity matrix for the element.

        Similar to :class:`Q4Element_K`, the matrix is assembled numerically
        using 2x2 Gauss quadrature.  This removes the dependency on SymPy for
        runtime calculations and considerably reduces initialization time.

        Returns
        -------
        numpy.ndarray
            Conductivity matrix of shape ``(4, 4)``.

        Examples
        --------
        >>> from tofea.elements import Q4Element_T
        >>> Q4Element_T().element.shape
        (4, 4)
        """

        C: NDArray = np.array([[self.k, 0], [0, self.k]], dtype=self.dtype)

        K: NDArray = np.zeros((4, 4), dtype=self.dtype)

        for xi, eta in self.gauss_points():
            dN_dxi, dN_deta = self.grad_shape_funcs(xi, eta)

            B: NDArray = np.zeros((2, 4), dtype=self.dtype)
            for i in range(4):
                B[0, i] = dN_dxi[i] / self.dx
                B[1, i] = dN_deta[i] / self.dy

            K += (B.T @ C @ B) * self.dx * self.dy

        K[np.abs(K) < self.eps] = 0
        return K
