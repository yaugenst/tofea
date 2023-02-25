from functools import cached_property
from pathlib import Path
from time import time

import h5py
import numpy as np
from autograd.extend import defvjp, primitive
from pyMKL import pardisoSolver
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

from .elements import Q4Element_K, Q4Element_T


class FEM2D_K:
    def __init__(
        self, domain, dofs, fixed, load, solver="direct", element="./data/Q4_K.h5",
    ):
        self.solver = solver

        self.shape = domain
        self.forces = load.ravel()
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self.element_stiffness = self.load_element(element)
        self.displacement = np.zeros(dofs.size)

        defvjp(self.compliance, self.compliance_vjp)

    @staticmethod
    def load_element(fp):
        fp = Path(fp)
        if fp.is_file():
            with h5py.File(fp, "r") as f:
                K = np.array(f["K"])
        else:
            fp.parent.mkdir(exist_ok=True, parents=True)
            K = Q4Element_K().get_element()
            with h5py.File(fp, "w") as f:
                f.create_dataset("K", data=K, compression="gzip")
        return K

    @cached_property
    def dofmap(self):
        nelx, nely = self.shape
        b = np.arange(2 * (nely + 1), 2 * (nely + 1) + 2)
        a = b + 2
        return np.r_[2, 3, a, b, 0, 1]

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

    def global_stiffness(self, x):
        i, j = np.meshgrid(range(len(self.dofmap)), range(len(self.dofmap)))
        nelx, nely = x.shape

        vals, ix, iy = [], [], []
        for elx in range(nelx):
            for ely in range(nely):
                e2sdofmap = self.dofmap + x.ndim * (ely + elx * (nely + 1))
                v = x[elx, ely] * self.element_stiffness
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
    def compliance(x, self):
        nelx, nely = x.shape
        X, Y = np.indices(x.shape)
        e2sdofmap = np.expand_dims(self.dofmap.reshape(-1, 1), axis=1)
        e2sdofmap = np.add(e2sdofmap, x.ndim * (Y + X * (nely + 1)))

        print("assemble matrix...", end=" ")
        t0 = time()
        Kfree = self.global_stiffness(x)
        print(f"done: {time() - t0:.3f}s")

        if self.solver == "direct":
            print("direct solve...", end=" ")
            t0 = time()
            solver = pardisoSolver(Kfree, mtype=2)
            self.displacement[self.freedofs] = solver.run_pardiso(
                phase=13, rhs=self.forces[self.freedofs]
            )
            solver.clear()
            print(f"done: {time() - t0:.3f}s")
        elif self.solver == "iterative":
            print("iterative solve...", end=" ")
            t0 = time()
            self.displacement[self.freedofs], _ = cg(Kfree, self.forces[self.freedofs])
            print(f"done: {time() - t0:.3f}s")
        else:
            raise RuntimeError(f"Unknown solver: {self.solver}")

        Qe = self.displacement[e2sdofmap]
        QeK = np.tensordot(Qe, self.element_stiffness, axes=(0, 0))
        Qe_T = np.swapaxes(Qe, 2, 1).T
        self._QeKQe = np.einsum("mnk,mnk->mn", QeK, Qe_T)
        return np.sum(x * self._QeKQe)

    @staticmethod
    def compliance_vjp(ans, x, self):
        del ans, x

        def vjp(g):
            return -g * self._QeKQe

        return vjp

    def __call__(self, x):
        return self.compliance(x, self)


class FEM2D_T:
    def __init__(
        self, domain, dofs, fixed, load, solver="direct", element="./data/Q4_T.h5",
    ):
        self.solver = solver

        self.shape = domain
        self.forces = load.ravel()
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self.element_stiffness = self.load_element(element)
        self.displacement = np.zeros(dofs.size)

        defvjp(self.compliance, self.compliance_vjp)

    @staticmethod
    def load_element(fp):
        fp = Path(fp)
        if fp.is_file():
            with h5py.File(fp, "r") as f:
                K = np.array(f["K"])
        else:
            fp.parent.mkdir(exist_ok=True, parents=True)
            K = Q4Element_T().get_element()
            with h5py.File(fp, "w") as f:
                f.create_dataset("K", data=K, compression="gzip")
        return K

    @cached_property
    def dofmap(self):
        nelx, nely = self.shape
        return np.r_[1, (nely + 2), (nely + 1), 0]

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

    def global_stiffness(self, x):
        i, j = np.meshgrid(range(len(self.dofmap)), range(len(self.dofmap)))
        nelx, nely = x.shape

        vals, ix, iy = [], [], []
        for elx in range(nelx):
            for ely in range(nely):
                e2sdofmap = self.dofmap + (ely + elx * (nely + 1))
                v = x[elx, ely] * self.element_stiffness
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
    def compliance(x, self):
        nelx, nely = x.shape
        X, Y = np.indices(x.shape)
        e2sdofmap = np.expand_dims(self.dofmap.reshape(-1, 1), axis=1)
        e2sdofmap = np.add(e2sdofmap, Y + X * (nely + 1))

        print("assemble matrix...", end=" ")
        t0 = time()
        Kfree = self.global_stiffness(x)
        print(f"done: {time() - t0:.3f}s")

        if self.solver == "direct":
            print("direct solve...", end=" ")
            t0 = time()
            solver = pardisoSolver(Kfree, mtype=2)
            self.displacement[self.freedofs] = solver.run_pardiso(
                phase=13, rhs=self.forces[self.freedofs]
            )
            solver.clear()
            print(f"done: {time() - t0:.3f}s")
        elif self.solver == "iterative":
            print("iterative solve...", end=" ")
            t0 = time()
            self.displacement[self.freedofs], _ = cg(Kfree, self.forces[self.freedofs])
            print(f"done: {time() - t0:.3f}s")
        else:
            raise RuntimeError(f"Unknown solver: {self.solver}")

        Qe = self.displacement[e2sdofmap]
        QeK = np.tensordot(Qe, self.element_stiffness, axes=(0, 0))
        Qe_T = np.swapaxes(Qe, 2, 1).T
        self._QeKQe = np.einsum("mnk,mnk->mn", QeK, Qe_T)
        return np.sum(x * self._QeKQe)

    @staticmethod
    def compliance_vjp(ans, x, self):
        del ans, x

        def vjp(g):
            return -g * self._QeKQe

        return vjp

    def __call__(self, x):
        return self.compliance(x, self)
