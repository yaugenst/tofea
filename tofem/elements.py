from functools import cached_property

import numpy as np
import sympy

__all__ = ["Q4Element_K", "Q4Element_T", "H8Element_K", "H8Element_T"]


class Element:
    _dx, _dy, _dz = 0.5, 0.5, 0.5  # dimensions
    _eps = 1e-6

    @staticmethod
    def _b_entries(rule, shape_funcs, clist):
        shape_list = np.concatenate([x * np.asarray(rule) for x in shape_funcs])
        return tuple(map(sympy.diff, shape_list, clist))

    def shape_funcs(self):
        raise NotImplementedError

    def get_element(self):
        raise NotImplementedError


class Q4Element(Element):
    @cached_property
    def symbols(self):
        return sympy.symbols("a b x y", real=True)

    @cached_property
    def shape_funcs(self):
        a, b, x, y = self.symbols
        return [
            (a - x) * (b - y) / (4 * a * b),
            (a + x) * (b - y) / (4 * a * b),
            (a + x) * (b + y) / (4 * a * b),
            (a - x) * (b + y) / (4 * a * b),
        ]


class Q4Element_K(Q4Element):
    def __init__(self, e=1.0, nu=1 / 3):
        self.e = e
        self.nu = nu

    def get_element(self):
        a, b, x, y = self.symbols
        E, nu = sympy.symbols("E nu", real=True)

        B = sympy.Matrix(
            [
                self._b_entries([1, 0], self.shape_funcs, 8 * [x]),
                self._b_entries([0, 1], self.shape_funcs, 8 * [y]),
                self._b_entries([1, 1], self.shape_funcs, 4 * [y, x]),
            ]
        )

        C = (E / (1 - nu ** 2)) * sympy.Matrix(
            [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
        )

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b))
        K = np.array(
            K.subs({a: self._dx, b: self._dy, E: self.e, nu: self.nu}),
            dtype=np.float64,
        )
        K[np.abs(K) < self._eps] = 0
        return K


class Q4Element_T(Q4Element):
    def __init__(self, k=1.0):
        self.k = k

    def get_element(self):
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
        K = np.array(K.subs({a: self._dx, b: self._dy, k: self.k}), dtype=np.float64)
        K[np.abs(K) < self._eps] = 0
        return K


class H8Element(Element):
    @cached_property
    def symbols(self):
        return sympy.symbols("a b c x y z", real=True)

    @cached_property
    def shape_funcs(self):
        a, b, c, x, y, z = self.symbols
        return [
            (a - x) * (b - y) * (c - z) / (8 * a * b * c),
            (a + x) * (b - y) * (c - z) / (8 * a * b * c),
            (a + x) * (b + y) * (c - z) / (8 * a * b * c),
            (a - x) * (b + y) * (c - z) / (8 * a * b * c),
            (a - x) * (b - y) * (c + z) / (8 * a * b * c),
            (a + x) * (b - y) * (c + z) / (8 * a * b * c),
            (a + x) * (b + y) * (c + z) / (8 * a * b * c),
            (a - x) * (b + y) * (c + z) / (8 * a * b * c),
        ]


class H8Element_K(H8Element):
    def __init__(self, e=1.0, nu=1 / 3):
        self.e = e
        self.nu = nu
        self.G = e / (2 * (1 + nu))
        self.g = e / ((1 + nu) * (1 - 2 * nu))

    def get_element(self):
        a, b, c, x, y, z = self.symbols
        E, nu, g, G = sympy.symbols("E nu g G", real=True)
        o = sympy.symbols("o", real=True)  # dummy symbol

        B = sympy.Matrix(
            [
                self._b_entries([1, 0, 0], self.shape_funcs, 24 * [x]),
                self._b_entries([0, 1, 0], self.shape_funcs, 24 * [y]),
                self._b_entries([0, 0, 1], self.shape_funcs, 24 * [z]),
                self._b_entries([1, 1, 1], self.shape_funcs, 8 * [y, x, o]),
                self._b_entries([1, 1, 1], self.shape_funcs, 8 * [o, z, y]),
                self._b_entries([1, 1, 1], self.shape_funcs, 8 * [z, o, x]),
            ]
        )

        C = sympy.Matrix(
            [
                [(1 - nu) * g, nu * g, nu * g, 0, 0, 0],
                [nu * g, (1 - nu) * g, nu * g, 0, 0, 0],
                [nu * g, nu * g, (1 - nu) * g, 0, 0, 0],
                [0, 0, 0, G, 0, 0],
                [0, 0, 0, 0, G, 0],
                [0, 0, 0, 0, 0, G],
            ]
        )

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b), (z, -c, c))
        K = np.array(
            K.subs(
                {
                    a: self._dx,
                    b: self._dy,
                    c: self._dz,
                    E: self.e,
                    nu: self.nu,
                    g: self.g,
                    G: self.G,
                }
            ),
            dtype=np.float64,
        )
        K[np.abs(K) < self._eps] = 0
        return K


class H8Element_T(H8Element):
    def __init__(self, k=1.0):
        self.k = k

    def get_element(self):
        a, b, c, x, y, z = self.symbols
        k = sympy.symbols("k")

        B = sympy.Matrix(
            [
                self._b_entries([1], self.shape_funcs, 8 * [x]),
                self._b_entries([1], self.shape_funcs, 8 * [y]),
                self._b_entries([1], self.shape_funcs, 8 * [z]),
            ]
        )

        C = sympy.Matrix([[k, 0, 0], [0, k, 0], [0, 0, k]])

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b), (z, -c, c))
        K = np.array(
            K.subs({a: self._dx, b: self._dy, c: self._dz, k: self.k}),
            dtype=np.float64,
        )
        K[np.abs(K) < self._eps] = 0
        return K
