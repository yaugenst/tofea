"""Helper utilities for defining boundary conditions and loads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np

__all__ = ["BoundaryConditions"]


def _edge_indices(edge: str) -> tuple:
    match edge:
        case "left":
            return 0, slice(None)
        case "right":
            return -1, slice(None)
        case "bottom":
            return slice(None), 0
        case "top":
            return slice(None), -1
        case _:
            raise ValueError(f"Invalid edge name: {edge}")


@dataclass
class BoundaryConditions:
    """Container for fixed degrees of freedom and loads.

    Parameters
    ----------
    shape:
        Number of elements in ``x`` and ``y`` direction.
    dof_dim:
        Degrees of freedom per node.

    Examples
    --------
    >>> bc = BoundaryConditions((1, 1))
    >>> bc.fix_edge("left")
    >>> bool(bc.fixed[0, 0])
    True
    >>> bc.apply_point_load(1, 1, 2.0)
    >>> float(bc.load[1, 1])
    2.0
    """

    shape: tuple[int, int]
    dof_dim: int = 1

    def __post_init__(self) -> None:
        node_shape = (self.shape[0] + 1, self.shape[1] + 1)
        if self.dof_dim == 1:
            self.fixed = np.zeros(node_shape, dtype="?")
            self.load = np.zeros(node_shape, dtype=float)
        else:
            self.fixed = np.zeros((*node_shape, self.dof_dim), dtype="?")
            self.load = np.zeros((*node_shape, self.dof_dim), dtype=float)

    def fix_edge(self, edge: str) -> None:
        """Fix all nodes along an edge."""
        idx = _edge_indices(edge)
        if self.dof_dim == 1:
            self.fixed[idx] = True
        else:
            self.fixed[(*idx, slice(None))] = True

    def apply_point_load(
        self, node_x: int, node_y: int, load_vector: float | Iterable[float]
    ) -> None:
        """Apply a load vector at a single node."""
        if self.dof_dim == 1:
            self.load[node_x, node_y] = float(cast(float, load_vector))
        else:
            vec = np.asarray(tuple(cast(Iterable[float], load_vector)), dtype=float)
            if vec.size != self.dof_dim:
                raise ValueError("load_vector has wrong size")
            self.load[node_x, node_y] = vec

    def apply_uniform_load_on_edge(self, edge: str, load_vector: float | Iterable[float]) -> None:
        """Apply the same load to all nodes on an edge."""
        idx = _edge_indices(edge)
        if self.dof_dim == 1:
            self.load[idx] = float(cast(float, load_vector))
        else:
            vec = np.asarray(tuple(cast(Iterable[float], load_vector)), dtype=float)
            if vec.size != self.dof_dim:
                raise ValueError("load_vector has wrong size")
            self.load[(*idx, slice(None))] = vec
