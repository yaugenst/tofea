"""Finite element primitives used by :mod:`tofea`."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import sympy
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class Element:
    """Base dataclass for finite elements."""

    dx: float = 0.5
    dy: float = 0.5
    dz: float = 0.5
    eps: float = 1e-6
    dtype: type = np.float64

    @staticmethod
    def _b_entries(
        rule: Iterable[int],
        shape_funcs: Iterable[sympy.Expr],
        clist: Iterable[sympy.Symbol],
    ) -> tuple[sympy.Expr, ...]:
        """Return derivatives of ``shape_funcs`` with respect to ``clist``.

        Parameters
        ----------
        rule
            Scaling applied to each shape function before differentiation.
        shape_funcs
            Iterable containing the shape functions.
        clist
            Symbols with respect to which the derivatives are taken.

        Returns
        -------
        tuple[sympy.Expr, ...]
            The differentiated shape functions.

        Examples
        --------
        >>> import sympy
        >>> x, y = sympy.symbols("x y")
        >>> Element._b_entries([1, 0], [x * y], [x, y])
        (y, 0)
        """

        shape_list = np.concatenate([x * np.asarray(rule) for x in shape_funcs])
        return tuple(map(sympy.diff, shape_list, clist))


class Q4Element(Element):
    """Four-node quadrilateral element."""

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Return the symbols used to define the element."""

        return sympy.symbols("a b x y", real=True)

    @property
    def shape_funcs(self) -> list[sympy.Expr]:
        """Return bilinear shape functions for a four-node element."""

        a, b, x, y = self.symbols
        return [
            (a - x) * (b - y) / (4 * a * b),
            (a + x) * (b - y) / (4 * a * b),
            (a + x) * (b + y) / (4 * a * b),
            (a - x) * (b + y) / (4 * a * b),
        ]


@dataclass(frozen=True, slots=True)
class Q4Element_K(Q4Element):
    """Plane stress elasticity element."""

    e: float = 1.0
    nu: float = 1 / 3

    @cached_property
    def element(self) -> NDArray:
        """Return the 8x8 stiffness matrix for the element.

        The matrix is computed symbolically and evaluated for the instance
        parameters.  Small values below ``eps`` are zeroed out to keep the
        matrix well conditioned.

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

        a, b, x, y = self.symbols
        E, nu = sympy.symbols("E nu", real=True)

        B = sympy.Matrix(
            [
                self._b_entries([1, 0], self.shape_funcs, 8 * [x]),
                self._b_entries([0, 1], self.shape_funcs, 8 * [y]),
                self._b_entries([1, 1], self.shape_funcs, 4 * [y, x]),
            ]
        )

        C = (E / (1 - nu**2)) * sympy.Matrix(
            [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
        )

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b))
        K = np.array(
            K.subs({a: self.dx, b: self.dy, E: self.e, nu: self.nu}),
            dtype=self.dtype,
        )
        K[np.abs(K) < self.eps] = 0
        return K


@dataclass(frozen=True, slots=True)
class Q4Element_T(Q4Element):
    """Heat conductivity element."""

    k: float = 1.0

    @cached_property
    def element(self) -> NDArray:
        """Return the 4x4 conductivity matrix for the element.

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

        a, b, x, y = self.symbols
        k = sympy.symbols("k", real=True)

        B = sympy.Matrix(
            [
                self._b_entries([1], self.shape_funcs, 4 * [x]),
                self._b_entries([1], self.shape_funcs, 4 * [y]),
            ]
        )

        C = sympy.Matrix([[k, 0], [0, k]])

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b))
        K = np.array(K.subs({a: self.dx, b: self.dy, k: self.k}), dtype=self.dtype)
        K[np.abs(K) < self.eps] = 0
        return K
