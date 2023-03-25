#!/usr/bin/env python3

import autograd.numpy as np
import matplotlib.pyplot as plt
import nlopt
from autograd import tensor_jacobian_product, value_and_grad
from matplotlib.colors import CenteredNorm

from tofea.fea2d import FEA2D_K
from tofea.topopt_helpers import simp_parametrization

max_its = 100
volfrac = 0.4
sigma = 1.0
shape = (160, 80)
nelx, nely = shape
emin, emax = 1e-4, 1

dofs = np.arange(2 * (nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1, 2)
fixed = np.zeros_like(dofs, dtype=bool)
load = np.zeros_like(dofs)

fixed[0, :, :] = 1
load[-1, -1, 1] = 1

fem = FEA2D_K(fixed, load)
parametrization = simp_parametrization(shape, sigma, emin, emax)
x0 = np.full(shape, volfrac)

plt.ion()
fig, ax = plt.subplots(4, 1)
im1 = ax[0].imshow(parametrization(x0).T, cmap="gray_r", vmin=emin, vmax=emax)
ax[1].imshow(np.zeros_like(x0), cmap="magma")
ax[2].imshow(np.zeros_like(load[..., 0]), cmap="RdBu", norm=CenteredNorm())
ax[3].imshow(np.zeros_like(load[..., 1]), cmap="RdBu", norm=CenteredNorm())
fig.tight_layout()


def volume_constraint(x, gd):
    v, g = value_and_grad(np.mean)(x)
    if gd.size > 0:
        gd[:] = g
    return v - volfrac


def nlopt_obj(x, gd):
    design = parametrization(x)
    c, dc = value_and_grad(fem)(design)

    if gd.size > 0:
        gd[:] = tensor_jacobian_product(parametrization)(x, dc).ravel()

    im1.set_data(design.T)
    ax[1].cla()
    ax[1].imshow(
        design.T * np.reshape(fem._u, load.shape)[1:, 1:, 0].T,
        cmap="RdBu",
        norm=CenteredNorm(),
    )
    ax[2].cla()
    ax[2].imshow(
        design.T * np.reshape(fem._u, load.shape)[1:, 1:, 1].T,
        cmap="RdBu",
        norm=CenteredNorm(),
    )
    ax[3].cla()
    ax[3].imshow(-fem._c.T, cmap="magma")
    plt.pause(0.01)

    return c


opt = nlopt.opt(nlopt.LD_MMA, x0.size)
opt.add_inequality_constraint(volume_constraint)
opt.set_min_objective(nlopt_obj)
opt.set_lower_bounds(0)
opt.set_upper_bounds(1)
opt.set_maxeval(max_its)
opt.optimize(x0.ravel())

plt.show(block=True)
