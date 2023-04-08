from functools import cached_property

import autograd.numpy as anp
import numpy as np
from numpy.typing import NDArray

from tofea.elements import Q4Element_K, Q4Element_T
from tofea.primitives import solve_coo


class FEA2D:
    def __init__(self, fixed: NDArray[np.bool_], solver: str = "scipy") -> None:
        nx, ny = fixed.shape[:2]
        dofs = np.arange(fixed.size, dtype=np.uint32).reshape(fixed.shape)
        self.out_shape = (nx - 1, ny - 1)
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self.solver = solver

    @cached_property
    def index_map(self) -> NDArray[np.uint32]:
        indices = np.concatenate([self.freedofs, self.fixdofs])
        imap = np.zeros(len(indices), dtype=np.uint32)
        imap[indices] = np.arange(len(indices), dtype=np.uint32)
        return imap

    @cached_property
    def e2sdofmap(self) -> NDArray[np.uint32]:
        nx, ny = self.out_shape
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

    def __call__(self, x: NDArray, b: NDArray) -> float:
        data, indices = self.global_mat(x)
        u_nz = solve_coo(data, indices, b.ravel()[self.freedofs], self.solver)
        u = anp.concatenate([u_nz, np.zeros(len(self.fixdofs))])[self.index_map]

        dofmap = np.reshape(self.e2sdofmap.T, (-1, *self.out_shape))
        c = anp.einsum("ixy,ij,jxy->xy", u[dofmap], self.element, u[dofmap])

        return anp.sum(c)


class FEA2D_K(FEA2D):
    dof_dim: int = 2

    @cached_property
    def element(self) -> NDArray:
        return Q4Element_K().element

    @cached_property
    def dofmap(self) -> NDArray[np.uint32]:
        _, nely = self.out_shape
        b = np.arange(2 * (nely + 1), 2 * (nely + 1) + 2)
        a = b + 2
        return np.r_[2, 3, a, b, 0, 1].astype(np.uint32)


class FEA2D_T(FEA2D):
    dof_dim: int = 1

    @cached_property
    def element(self) -> NDArray:
        return Q4Element_T().element

    @cached_property
    def dofmap(self) -> NDArray[np.uint32]:
        _, nely = self.out_shape
        return np.r_[1, (nely + 2), (nely + 1), 0].astype(np.uint32)
