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

    @staticmethod
    def inverse_permutation(indices):
        inverse_perm = np.zeros(len(indices), dtype=np.int64)
        inverse_perm[indices] = np.arange(len(indices), dtype=np.int64)
        return inverse_perm

    def _get_dof_indices(self, k_ylist, k_xlist):
        index_map = self.inverse_permutation(
            np.concatenate([self.freedofs, self.fixdofs])
        )
        keep = np.isin(k_xlist, self.freedofs) & np.isin(k_ylist, self.freedofs)
        i = index_map[k_ylist][keep]
        j = index_map[k_xlist][keep]
        return index_map, keep, np.stack([i, j])

    def global_mat(self, x):
        i, j = np.meshgrid(range(len(self.dofmap)), range(len(self.dofmap)))
        nelx, nely = x.shape

        vals, ix, iy = [], [], []
        for elx in range(nelx):
            for ely in range(nely):
                e2sdofmap = self.dofmap + self.dof_dim * (ely + elx * (nely + 1))
                v = x[elx, ely] * self.element
                ix.append(e2sdofmap[i])
                iy.append(e2sdofmap[j])
                vals.append(v[i, j])
        vals = np.asarray(vals).ravel()
        ix = np.asarray(ix, dtype=int).ravel()
        iy = np.asarray(iy, dtype=int).ravel()
        _, keep, indices = self._get_dof_indices(ix, iy)
        return coo_matrix((vals[keep], indices)).tocsr()

    @staticmethod
    @primitive
    def fea(x, self):
        X, Y = np.indices(x.shape)
        e2sdofmap = np.expand_dims(self.dofmap.reshape(-1, 1), axis=1)
        e2sdofmap = np.add(e2sdofmap, Y + X * (x.shape[1] + 1))

        system = self.global_mat(x)
        self.fea_result[self.freedofs] = spsolve(system, self.load[self.freedofs])

        Qe = self.fea_result[e2sdofmap]
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
        return np.r_[2, 3, a, b, 0, 1]


class FEA2D_T(FEA2D):
    dof_dim = 1

    @cached_property
    def element(self):
        return Q4Element_T().get_element()

    @cached_property
    def dofmap(self):
        _, nely = self.shape
        return np.r_[1, (nely + 2), (nely + 1), 0]
