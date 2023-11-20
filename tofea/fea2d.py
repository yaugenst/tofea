from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import autograd.numpy as anp
import numpy as np
from numpy.typing import NDArray

from tofea import DEFAULT_SOLVER
from tofea.elements import Q4Element_K, Q4Element_T
from tofea.primitives import solve_coo
from tofea.solvers import Solver, get_solver


@dataclass
class FEA2D(ABC):
    fixed: NDArray[np.bool_]
    solver: str = DEFAULT_SOLVER
    dx: float = 0.5
    dy: float = 0.5

    @property
    @abstractmethod
    def dof_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def element(self) -> NDArray:
        ...

    @property
    @abstractmethod
    def dofmap(self) -> NDArray[np.uint32]:
        ...

    @property
    def shape(self) -> tuple[int, int]:
        nx, ny = self.fixed.shape[:2]
        return nx - 1, ny - 1

    @cached_property
    def dofs(self) -> NDArray[np.uint32]:
        return np.arange(self.fixed.size, dtype=np.uint32)

    @cached_property
    def fixdofs(self) -> NDArray[np.uint32]:
        return self.dofs[self.fixed.ravel()]

    @cached_property
    def freedofs(self) -> NDArray[np.uint32]:
        return self.dofs[~self.fixed.ravel()]

    @cached_property
    def _solver(self) -> Solver:
        return get_solver(self.solver)

    @cached_property
    def index_map(self) -> NDArray[np.uint32]:
        indices = np.concatenate([self.freedofs, self.fixdofs])
        imap = np.zeros_like(self.dofs)
        imap[indices] = self.dofs
        return imap

    @cached_property
    def e2sdofmap(self) -> NDArray[np.uint32]:
        nx, ny = self.shape
        x, y = np.unravel_index(np.arange(nx * ny), (nx, ny))
        idxs = self.dof_dim * (y + x * (ny + 1))
        return np.add(self.dofmap[None], idxs[:, None].astype(np.uint32))

    @cached_property
    def keep_indices(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.uint32]]:
        i, j = np.meshgrid(range(len(self.dofmap)), range(len(self.dofmap)))
        ix = self.e2sdofmap[:, i].ravel()
        iy = self.e2sdofmap[:, j].ravel()
        keep = np.isin(ix, self.freedofs) & np.isin(iy, self.freedofs)
        indices = np.stack([self.index_map[ix][keep], self.index_map[iy][keep]])
        return keep, indices

    def global_mat(self, x: NDArray) -> tuple[NDArray, NDArray]:
        x = np.reshape(x, (-1, 1, 1)) * self.element[None]
        x = x.ravel()
        keep, indices = self.keep_indices
        return x[keep], indices

    def solve(self, x: NDArray, b: NDArray) -> NDArray:
        data, indices = self.global_mat(x)
        u_nz = solve_coo(data, indices, b.ravel()[self.freedofs], self._solver)
        u = anp.concatenate([u_nz, np.zeros(len(self.fixdofs))])[self.index_map]
        return u


@dataclass
class FEA2D_K(FEA2D):
    dof_dim: int = 2
    e: float = 1.0
    nu: float = 1 / 3

    @cached_property
    def element(self) -> NDArray:
        return Q4Element_K(e=self.e, nu=self.nu, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> NDArray[np.uint32]:
        _, nely = self.shape
        b = np.arange(2 * (nely + 1), 2 * (nely + 1) + 2)
        a = b + 2
        return np.r_[2, 3, a, b, 0, 1].astype(np.uint32)

    def displacement(self, x: NDArray, b: NDArray) -> NDArray:
        return self.solve(x, b)

    def compliance(self, x: NDArray, displacement: NDArray) -> NDArray:
        dofmap = np.reshape(self.e2sdofmap.T, (-1, *self.shape))
        c = anp.einsum(
            "ixy,ij,jxy->xy", displacement[dofmap], self.element, displacement[dofmap]
        )
        return anp.sum(x * c)


@dataclass
class FEA2D_T(FEA2D):
    dof_dim: int = 1
    k: float = 1.0

    @cached_property
    def element(self) -> NDArray:
        return Q4Element_T(k=self.k, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> NDArray[np.uint32]:
        _, nely = self.shape
        return np.r_[1, (nely + 2), (nely + 1), 0].astype(np.uint32)

    def temperature(self, x: NDArray, b: NDArray) -> NDArray:
        return self.solve(x, b)
