from functools import cached_property

import numpy as np
from autograd.extend import defvjp, primitive
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .elements import Q4Element_K, Q4Element_T


class FEA2D:
    def __init__(self, fixed: NDArray[np.bool_], load: NDArray[np.bool_]) -> None:
        nx, ny = fixed.shape[:2]
        dofs = np.arange(fixed.size, dtype=np.uint32).reshape(fixed.shape)
        self.out_shape = (nx - 1, ny - 1)
        self.load = load.ravel()
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self._u = np.zeros(dofs.size)
        self._c = np.zeros(self.out_shape)

        defvjp(self.fea, self.fea_vjp)

    @cached_property
    def index_map(self) -> NDArray[np.uint32]:
        indices = np.concatenate([self.freedofs, self.fixdofs])
        imap = np.zeros(len(indices), dtype=np.uint32)
        imap[indices] = np.arange(len(indices), dtype=np.uint32)
        return imap

    @cached_property
    def e2sdofmap(self) -> NDArray[np.uint32]:
        nx, ny = self.out_shape
        idxs = np.arange(nx * ny, dtype=np.uint32)
        return np.add(
            self.dofmap[None],
            (self.dof_dim * (idxs % ny + idxs // ny * (ny + 1)))[:, None],
        )

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

    def global_mat(self, x: NDArray) -> csr_matrix:
        x = np.reshape(x, (-1, 1, 1)) * self.element[None]
        x = x.ravel()
        keep, indices = self.keep_indices
        return coo_matrix((x[keep], indices)).tocsr()

    @staticmethod
    @primitive
    def fea(x: NDArray, self) -> float:
        system = self.global_mat(x)
        dm = np.reshape(self.e2sdofmap.T, (-1, *self.out_shape))
        self._u[self.freedofs] = spsolve(system, self.load[self.freedofs])
        self._c[:] = np.einsum("ixy,ij,jxy->xy", self._u[dm], self.element, self._u[dm])
        return np.sum(x * self._c)

    @staticmethod
    def fea_vjp(ans: float, x: NDArray, self) -> NDArray:
        return lambda g: -g * self._c

    def __call__(self, x: NDArray) -> float:
        return self.fea(x, self)


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
