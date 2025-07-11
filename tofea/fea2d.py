"""Finite element solvers for 2D problems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import autograd.numpy as anp
import numpy as np
from numpy.typing import NDArray

from tofea import DEFAULT_SOLVER
from tofea.elements import Q4Element_K, Q4Element_T
from tofea.primitives import solve_and_compute_self_adjoint_objective, solve_coo
from tofea.solvers import Solver, get_solver


@dataclass
class FEA2D(ABC):
    """Abstract base class for 2D finite element models.

    Parameters
    ----------
    fixed
        Boolean mask indicating which degrees of freedom are fixed.
        Shape for thermal problems is ``(nx + 1, ny + 1)`` and for
        elasticity problems ``(nx + 1, ny + 1, 2)``.
    solver
        Name of the backend solver to use.
    dx, dy
        Element dimensions in ``x`` and ``y`` direction.

    Notes
    -----
    Subclasses must provide the element matrix and a mapping from element
    degrees of freedom to system degrees of freedom.
    """

    fixed: NDArray[np.bool_]
    solver: str = DEFAULT_SOLVER
    dx: float = 0.5
    dy: float = 0.5

    @property
    @abstractmethod
    def dof_dim(self) -> int: ...

    @property
    @abstractmethod
    def element(self) -> NDArray: ...

    @property
    @abstractmethod
    def dofmap(self) -> NDArray[np.uint32]: ...

    @property
    def shape(self) -> tuple[int, int]:
        """Number of elements in :math:`x` and :math:`y` direction."""

        nx, ny = self.fixed.shape[:2]
        return nx - 1, ny - 1

    @cached_property
    def dofs(self) -> NDArray[np.uint32]:
        """Indices of all degrees of freedom."""

        return np.arange(self.fixed.size, dtype=np.uint32)

    @cached_property
    def fixdofs(self) -> NDArray[np.uint32]:
        """Indices of fixed degrees of freedom."""

        return self.dofs[self.fixed.ravel()]

    @cached_property
    def freedofs(self) -> NDArray[np.uint32]:
        """Indices of free degrees of freedom."""

        return self.dofs[~self.fixed.ravel()]

    @cached_property
    def _solver(self) -> Solver:
        """Backend solver instance."""

        return get_solver(self.solver)

    @cached_property
    def index_map(self) -> NDArray[np.uint32]:
        """Permutation that places free DOFs first."""

        indices = np.concatenate([self.freedofs, self.fixdofs])
        imap = np.zeros_like(self.dofs)
        imap[indices] = self.dofs
        return imap

    @cached_property
    def e2sdofmap(self) -> NDArray[np.uint32]:
        """Map each element to its system DOF indices."""

        nx, ny = self.shape
        x, y = np.unravel_index(np.arange(nx * ny), (nx, ny))
        idxs = self.dof_dim * (y + x * (ny + 1))
        return np.add(self.dofmap[None], idxs[:, None].astype(np.uint32))

    @cached_property
    def e2sdofmap_reshaped(self) -> NDArray[np.uint32]:
        """Element mapping reshaped for tensor operations."""

        return np.reshape(self.e2sdofmap.T, (-1, *self.shape))

    @cached_property
    def keep_indices(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.uint32]]:
        """Indices for assembling the global matrix."""

        i, j = np.meshgrid(range(len(self.dofmap)), range(len(self.dofmap)))
        ix = self.e2sdofmap[:, i].ravel()
        iy = self.e2sdofmap[:, j].ravel()
        keep = np.isin(ix, self.freedofs) & np.isin(iy, self.freedofs)
        indices = np.stack([self.index_map[ix][keep], self.index_map[iy][keep]])
        return keep, indices

    def global_mat(self, x: NDArray) -> tuple[NDArray, NDArray]:
        """Assemble global matrix from element data."""

        x = np.reshape(x, (-1, 1, 1)) * self.element[None]
        x = x.ravel()
        keep, indices = self.keep_indices
        return x[keep], indices

    def solve(self, x: NDArray, b: NDArray) -> NDArray:
        """Solve ``K(x) u = b`` for ``u``."""

        data, indices = self.global_mat(x)
        u_nz = solve_coo(data, indices, b.ravel()[self.freedofs], self._solver)
        u = anp.concatenate([u_nz, np.zeros(len(self.fixdofs))])[self.index_map]
        return u

    def _self_adjoint_objective(self, x: NDArray, b: NDArray) -> NDArray:
        """Internal method to compute a self-adjoint objective."""

        data, indices = self.global_mat(x)
        free_rhs = b.ravel()[self.freedofs]
        objective, _ = solve_and_compute_self_adjoint_objective(
            data, indices, free_rhs, self._solver
        )
        return objective


@dataclass
class FEA2D_K(FEA2D):
    """Finite element model for compliance problems.

    This model solves small deformation elasticity problems using bilinear
    quadrilateral elements.

    Parameters
    ----------
    e : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.

    Examples
    --------
    >>> import numpy as np
    >>> from tofea.fea2d import FEA2D_K
    >>> fixed = np.zeros((1, 1, 2), dtype=bool)
    >>> fem = FEA2D_K(fixed, e=210e9, nu=0.3)
    >>> fem.element.shape
    (8, 8)
    """

    dof_dim: int = 2
    e: float = 1.0
    nu: float = 1 / 3

    @cached_property
    def element(self) -> NDArray:
        """Element stiffness matrix."""
        return Q4Element_K(e=self.e, nu=self.nu, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> NDArray[np.uint32]:
        """Mapping of element DOFs to system DOFs."""
        _, nely = self.shape
        b = np.arange(2 * (nely + 1), 2 * (nely + 1) + 2)
        a = b + 2
        return np.r_[2, 3, a, b, 0, 1].astype(np.uint32)

    def displacement(self, x: NDArray, b: NDArray) -> NDArray:
        """Return displacement field for density ``x`` and load ``b``.

        Parameters
        ----------
        x : numpy.ndarray
            Density field of shape ``(nx, ny)``.
        b : numpy.ndarray
            Load array of shape ``(nx + 1, ny + 1, 2)``.

        This is a general-purpose method that returns the full displacement field,
        suitable for constructing arbitrary objective functions. If your
        objective is compliance minimization, using the ``compliance()`` method is
        significantly more performant.
        """

        return self.solve(x, b)

    def compliance(self, x: NDArray, b: NDArray) -> NDArray:
        """Computes the compliance objective using a highly efficient self-adjoint pathway.

        This method is optimized for compliance minimization and avoids a redundant
        adjoint solve during gradient computation. For constructing arbitrary objective
        functions based on the displacement field, use the :meth:`displacement` method
        instead.

        Parameters
        ----------
        x : numpy.ndarray
            Density field of shape ``(nx, ny)``.
        b : numpy.ndarray
            Load array of shape ``(nx + 1, ny + 1, 2)``.

        Returns
        -------
        numpy.ndarray
            The scalar compliance value.
        """

        return self._self_adjoint_objective(x, b)


@dataclass
class FEA2D_T(FEA2D):
    """Finite element model for heat conduction problems.

    Parameters
    ----------
    k : float
        Thermal conductivity of the material.

    Examples
    --------
    >>> import numpy as np
    >>> from tofea.fea2d import FEA2D_T
    >>> fixed = np.zeros((1, 1), dtype=bool)
    >>> fem = FEA2D_T(fixed, k=200.0)
    >>> fem.element.shape
    (4, 4)
    """

    dof_dim: int = 1
    k: float = 1.0

    @cached_property
    def element(self) -> NDArray:
        """Element conductivity matrix."""
        return Q4Element_T(k=self.k, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> NDArray[np.uint32]:
        """Mapping of element DOFs to system DOFs."""
        _, nely = self.shape
        return np.r_[1, (nely + 2), (nely + 1), 0].astype(np.uint32)

    def temperature(self, x: NDArray, b: NDArray) -> NDArray:
        """Return temperature field for density ``x`` and load ``b``.

        Parameters
        ----------
        x : numpy.ndarray
            Density field of shape ``(nx, ny)``.
        b : numpy.ndarray
            Heat load array of shape ``(nx + 1, ny + 1)``.

        This is a general-purpose method that returns the full temperature field,
        suitable for constructing arbitrary objective functions. If your objective
        is thermal compliance minimization, using the ``thermal_compliance()``
        method is significantly more performant.
        """

        return self.solve(x, b)

    def thermal_compliance(self, x: NDArray, b: NDArray) -> NDArray:
        """Computes the thermal compliance objective using a highly efficient self-adjoint pathway.

        This method is optimized for thermal compliance minimization and avoids a
        redundant adjoint solve during gradient computation. For constructing arbitrary
        objective functions based on the temperature field, use the :meth:`temperature`
        method instead.

        Parameters
        ----------
        x : numpy.ndarray
            Density field of shape ``(nx, ny)``.
        b : numpy.ndarray
            Heat load array of shape ``(nx + 1, ny + 1)``.

        Returns
        -------
        numpy.ndarray
            The scalar thermal compliance value.
        """

        return self._self_adjoint_objective(x, b)
