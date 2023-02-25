from functools import cached_property
from pathlib import Path
from time import time

import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import h5py
import numpy as np
from autograd.extend import defvjp, primitive
from pyMKL import pardisoSolver
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

from .elements import H8Element_K, H8Element_T


class FEM3D_K:
    def __init__(
        self,
        domain,
        dofs,
        fixed,
        load,
        z_chunks=10,
        solver="direct",
        element="./data/H8_K.h5",
    ):
        self.solver = solver

        self.shape = domain
        self.forces = load.ravel()
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self.element_stiffness = self.load_element(element)
        self.displacement = np.zeros(dofs.size)
        self.chunks = z_chunks

        defvjp(self.compliance, self.compliance_vjp)

    @staticmethod
    def load_element(fp):
        fp = Path(fp)
        if fp.is_file():
            with h5py.File(fp, "r") as f:
                K = np.array(f["K"])
        else:
            fp.parent.mkdir(exist_ok=True, parents=True)
            K = H8Element_K().get_element()
            with h5py.File(fp, "w") as f:
                f.create_dataset("K", data=K, compression="gzip")
        return K

    @cached_property
    def dofmap(self):
        nelx, nely, nelz = self.shape
        a = np.arange(3) + 3
        b = np.arange(3 * (nely + 2), 3 * (nely + 2) + 3)
        c = np.arange(3 * (nely + 1), 3 * (nely + 1) + 3)
        d = np.arange(3)
        e = np.arange(
            3 * ((nelx + 1) * (nely + 1) + 1), 3 * ((nelx + 1) * (nely + 1) + 1) + 3,
        )
        f = np.arange(
            3 * ((nelx + 1) * (nely + 1) + (nely + 2)),
            3 * ((nelx + 1) * (nely + 1) + (nely + 2)) + 3,
        )
        g = np.arange(
            3 * ((nelx + 1) * (nely + 1) + (nely + 1)),
            3 * ((nelx + 1) * (nely + 1) + (nely + 1)) + 3,
        )
        h = np.arange(3 * (nelx + 1) * (nely + 1), 3 * (nelx + 1) * (nely + 1) + 3)
        return np.r_[a, b, c, d, e, f, g, h]

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
        nelx, nely, nelz = x.shape

        mat = None
        for chunk in np.array_split(np.arange(nelz), self.chunks)[::-1]:
            vals, ix, iy = [], [], []
            for elz in chunk:
                for elx in range(nelx):
                    for ely in range(nely):
                        e2sdofmap = self.dofmap + x.ndim * (
                            ely + elx * (nely + 1) + elz * (nelx + 1) * (nely + 1)
                        )
                        v = x[elx, ely, elz] * self.element_stiffness
                        ix.append(e2sdofmap[i])
                        iy.append(e2sdofmap[j])
                        vals.append(v[i, j])
            vals = np.asarray(vals).ravel()
            ix = np.asarray(ix, dtype=int).ravel()
            iy = np.asarray(iy, dtype=int).ravel()

            _, keep, indices = self._get_dof_indices(ix, iy)
            if mat is None:
                mat = coo_matrix((vals[keep], indices)).tocsr()
            else:
                mat += coo_matrix((vals[keep], indices), shape=mat.shape).tocsr()
        return mat

    @staticmethod
    @primitive
    def compliance(x, self):
        nelx, nely, nelz = x.shape
        X, Y, Z = np.indices(x.shape)
        X *= nely + 1
        Z *= (nelx + 1) * (nely + 1)
        e2sdofmap = np.expand_dims(self.dofmap.reshape(-1, 1, 1), axis=1)
        e2sdofmap = np.add(e2sdofmap, x.ndim * (X + Y + Z))

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
        elif self.solver == "gpu":
            print("gpu solve...", end=" ")
            t0 = time()
            self.displacement[self.freedofs] = cp.asnumpy(
                cl.cg(cs.csr_matrix(Kfree), cp.array(self.forces[self.freedofs]))[0]
            )
            print(f"done: {time() - t0:.3f}s")
        else:
            raise RuntimeError(f"Unknown solver: {self.solver}")

        Qe = self.displacement[e2sdofmap]
        QeK = np.tensordot(Qe, self.element_stiffness, axes=(0, 0))
        Qe_T = np.swapaxes(Qe.T, 2, 0)
        self._QeKQe = np.einsum("klmn,klmn->klm", QeK, Qe_T)
        return np.sum(x * self._QeKQe)

    @staticmethod
    def compliance_vjp(ans, x, self):
        del ans, x

        def vjp(g):
            return -g * self._QeKQe

        return vjp

    def __call__(self, x):
        return self.compliance(x, self)


class FEM3D_T:
    def __init__(
        self,
        domain,
        dofs,
        fixed,
        load,
        z_chunks=10,
        solver="direct",
        element="./data/H8_T.h5",
    ):
        self.solver = solver

        self.shape = domain
        self.forces = load.ravel()
        self.fixdofs = dofs[fixed].ravel()
        self.freedofs = dofs[~fixed].ravel()
        self.element_stiffness = self.load_element(element)
        self.displacement = np.zeros(dofs.size)
        self.chunks = z_chunks

        defvjp(self.compliance, self.compliance_vjp)

    @staticmethod
    def load_element(fp):
        fp = Path(fp)
        if fp.is_file():
            with h5py.File(fp, "r") as f:
                K = np.array(f["K"])
        else:
            fp.parent.mkdir(exist_ok=True, parents=True)
            K = H8Element_T().get_element()
            with h5py.File(fp, "w") as f:
                f.create_dataset("K", data=K, compression="gzip")
        return K

    @cached_property
    def dofmap(self):
        nelx, nely, nelz = self.shape
        e2s = np.r_[1, (nely + 2), (nely + 1), 0]
        return np.r_[e2s, (e2s + (nelx + 1) * (nely + 1))]

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
        nelx, nely, nelz = x.shape

        mat = None
        for chunk in np.array_split(np.arange(nelz), self.chunks)[::-1]:
            vals, ix, iy = [], [], []
            for elz in chunk:
                for elx in range(nelx):
                    for ely in range(nely):
                        e2sdofmap = self.dofmap + (
                            ely + elx * (nely + 1) + elz * (nelx + 1) * (nely + 1)
                        )
                        v = x[elx, ely, elz] * self.element_stiffness
                        ix.append(e2sdofmap[i])
                        iy.append(e2sdofmap[j])
                        vals.append(v[i, j])
            vals = np.asarray(vals).ravel()
            ix = np.asarray(ix, dtype=int).ravel()
            iy = np.asarray(iy, dtype=int).ravel()

            _, keep, indices = self._get_dof_indices(ix, iy)
            if mat is None:
                mat = coo_matrix((vals[keep], indices)).tocsr()
            else:
                mat += coo_matrix((vals[keep], indices), shape=mat.shape).tocsr()
        return mat

    @staticmethod
    @primitive
    def compliance(x, self):
        nelx, nely, nelz = x.shape
        X, Y, Z = np.indices(x.shape)
        X *= nely + 1
        Z *= (nelx + 1) * (nely + 1)
        e2sdofmap = np.expand_dims(self.dofmap.reshape(-1, 1, 1), axis=1)
        e2sdofmap = np.add(e2sdofmap, X + Y + Z)

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
        elif self.solver == "gpu":
            print("gpu solve...", end=" ")
            t0 = time()
            self.displacement[self.freedofs] = cp.asnumpy(
                cl.cg(cs.csr_matrix(Kfree), cp.array(self.forces[self.freedofs]))[0]
            )
            print(f"done: {time() - t0:.3f}s")
        else:
            raise RuntimeError(f"Unknown solver: {self.solver}")

        Qe = self.displacement[e2sdofmap]
        QeK = np.tensordot(Qe, self.element_stiffness, axes=(0, 0))
        Qe_T = np.swapaxes(Qe.T, 2, 0)
        self._QeKQe = np.einsum("klmn,klmn->klm", QeK, Qe_T)
        return np.sum(x * self._QeKQe)

    @staticmethod
    def compliance_vjp(ans, x, self):
        del ans, x

        def vjp(g):
            return -g * self._QeKQe

        return vjp

    def __call__(self, x):
        return self.compliance(x, self)
