#!/usr/bin/env python3

import autograd.numpy as anp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from autograd import tensor_jacobian_product, value_and_grad

from tofea.fea2d import FEA2D_T
from tofea.topopt_helpers import simp_parametrization

max_its = 100
volfrac = 0.3
sigma = 1.0
shape = (100, 100)
nelx, nely = shape
cmin, cmax = 1.0, 1e6

fixed = np.zeros((nelx + 1, nely + 1), dtype="?")
load = np.zeros_like(fixed)

fixed[40:60, -1] = 1
load[:, 0] = 1

fem = FEA2D_T(fixed, load)
parametrization = simp_parametrization(shape, sigma, cmin, cmax)
x0 = np.full(shape, volfrac)

plt.ion()
fig, ax = plt.subplots(2, 1)
im1 = ax[0].imshow(parametrization(x0).T, cmap="gray_r", vmin=cmin, vmax=cmax)
ax[1].imshow(np.zeros_like(x0), cmap="magma")
fig.tight_layout()


def volume_constraint(x, gd):
    v, g = value_and_grad(anp.mean)(x)
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
    ax[1].imshow((design * fem._c).T, cmap="magma")
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
