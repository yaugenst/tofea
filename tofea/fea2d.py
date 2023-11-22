from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
from jax import Array

from tofea.elements import Q4Element_K, Q4Element_T
from tofea.primitives import solve_coo


@dataclass
class FEA2D(ABC):
    fixed: Array
    dx: float = 0.5
    dy: float = 0.5

    @property
    @abstractmethod
    def dof_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def element(self) -> Array:
        ...

    @property
    @abstractmethod
    def dofmap(self) -> Array:
        ...

    @property
    def shape(self) -> tuple[int, int]:
        nx, ny = self.fixed.shape[:2]
        return nx - 1, ny - 1

    @cached_property
    def dofs(self) -> Array:
        return jnp.arange(self.fixed.size, dtype=jnp.uint32)

    @cached_property
    def fixdofs(self) -> Array:
        return self.dofs[self.fixed.ravel()]

    @cached_property
    def freedofs(self) -> Array:
        return self.dofs[~self.fixed.ravel()]

    @cached_property
    def index_map(self) -> Array:
        indices = jnp.concatenate([self.freedofs, self.fixdofs])
        imap = jnp.zeros_like(self.dofs)
        imap = imap.at[indices].set(self.dofs)
        return imap

    @cached_property
    def e2sdofmap(self) -> Array:
        nx, ny = self.shape
        x, y = jnp.unravel_index(jnp.arange(nx * ny), (nx, ny))
        idxs = self.dof_dim * (y + x * (ny + 1))
        return jnp.add(self.dofmap[None], idxs[:, None].astype(jnp.uint32))

    @cached_property
    def keep_indices(
        self,
    ) -> tuple[Array, Array]:
        r = jnp.arange(self.dofmap.size)
        i, j = jnp.meshgrid(r, r)
        ix = self.e2sdofmap[:, i].ravel()
        iy = self.e2sdofmap[:, j].ravel()
        keep = jnp.isin(ix, self.freedofs) & jnp.isin(iy, self.freedofs)
        indices = jnp.stack([self.index_map[ix][keep], self.index_map[iy][keep]])
        return keep, indices

    def global_mat(self, x: Array) -> tuple[Array, Array]:
        x = jnp.reshape(x, (-1, 1, 1)) * self.element[None]
        x = x.ravel()
        keep, indices = self.keep_indices
        return x[keep], indices

    def solve(self, x: Array, b: Array) -> Array:
        data, indices = self.global_mat(x)
        u_nz = solve_coo(data, indices, b.ravel()[self.freedofs])
        z = jnp.zeros(self.fixdofs.size)
        u = jnp.concatenate([u_nz, z])[self.index_map]
        return u


@dataclass
class FEA2D_K(FEA2D):
    dof_dim: int = 2
    e: float = 1.0
    nu: float = 1 / 3

    @cached_property
    def element(self) -> Array:
        return Q4Element_K(e=self.e, nu=self.nu, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> Array:
        _, nely = self.shape
        b = jnp.arange(2 * (nely + 1), 2 * (nely + 1) + 2)
        a = b + 2
        return jnp.r_[2, 3, a, b, 0, 1].astype(jnp.uint32)

    def displacement(self, x: Array, b: Array) -> Array:
        return self.solve(x, b)

    def compliance(self, x: Array, displacement: Array) -> Array:
        dofmap = jnp.reshape(self.e2sdofmap.T, (-1, *self.shape))
        c = jnp.einsum(
            "ixy,ij,jxy->xy", displacement[dofmap], self.element, displacement[dofmap]
        )
        return jnp.sum(x * c)


@dataclass
class FEA2D_T(FEA2D):
    dof_dim: int = 1
    k: float = 1.0

    @cached_property
    def element(self) -> Array:
        return Q4Element_T(k=self.k, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> Array:
        _, nely = self.shape
        return jnp.r_[1, (nely + 2), (nely + 1), 0].astype(jnp.uint32)

    def temperature(self, x: Array, b: Array) -> Array:
        return self.solve(x, b)
