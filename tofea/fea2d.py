from functools import cached_property

import numpy as np
from autograd.extend import defvjp, primitive
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .elements import Q4Element_K, Q4Element_T


class FEA2D:
    def __init__(self, domain, dofs, fixed, load):
        self.shape = domain
        self.load = load.ravel()
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self.fea_result = np.zeros(dofs.size)

        defvjp(self.fea, self.fea_vjp)

    @cached_property
    def index_map(self):
        indices = np.concatenate([self.freedofs, self.fixdofs])
        imap = np.zeros(len(indices), dtype="i4")
        imap[indices] = np.arange(len(indices), dtype="i4")
        return imap

    @cached_property
    def e2sdofmap(self):
        nx, ny = self.shape
        idxs = np.arange(nx * ny, dtype="i4")
        return np.add(
            self.dofmap[None],
            (self.dof_dim * (idxs % ny + idxs // ny * (ny + 1)))[:, None],
        )

    @cached_property
    def keep_indices(self):
        i, j = np.meshgrid(range(len(self.dofmap)), range(len(self.dofmap)))
        ix = self.e2sdofmap[:, i].ravel()
        iy = self.e2sdofmap[:, j].ravel()
        keep = np.isin(ix, self.freedofs) & np.isin(iy, self.freedofs)
        indices = np.stack([self.index_map[ix][keep], self.index_map[iy][keep]])
        return keep, indices

    def global_mat(self, x):
        x = np.reshape(x, (-1, 1, 1)) * self.element[None]
        x = x.ravel()
        keep, indices = self.keep_indices
        return coo_matrix((x[keep], indices)).tocsr()

    @staticmethod
    @primitive
    def fea(x, self):
        system = self.global_mat(x)
        self.fea_result[self.freedofs] = spsolve(system, self.load[self.freedofs])

        dm = np.moveaxis(np.reshape(self.e2sdofmap, (*self.shape, -1)), -1, 0)
        Qe = self.fea_result[dm]
        QeK = np.tensordot(Qe, self.element, axes=(0, 0))
        Qe_T = np.swapaxes(Qe, 2, 1).T
        self._QeKQe = np.einsum("mnk,mnk->mn", QeK, Qe_T)
        return np.sum(x * self._QeKQe)

    @staticmethod
    def fea_vjp(ans, x, self):
        return lambda g: -g * self._QeKQe

    def __call__(self, x):
        return self.fea(x, self)


class FEA2D_K(FEA2D):
    dof_dim = 2

    @cached_property
    def element(self):
        return Q4Element_K().get_element()

    @cached_property
    def dofmap(self):
        _, nely = self.shape
        b = np.arange(2 * (nely + 1), 2 * (nely + 1) + 2)
        a = b + 2
        return np.r_[2, 3, a, b, 0, 1].astype("i4")


class FEA2D_T(FEA2D):
    dof_dim = 1

    @cached_property
    def element(self):
        return Q4Element_T().get_element()

    @cached_property
    def dofmap(self):
        _, nely = self.shape
        return np.r_[1, (nely + 2), (nely + 1), 0].astype("i4")
