#!/usr/bin/env python3

import autograd.numpy as np
import matplotlib.pyplot as plt
import nlopt
from autograd import value_and_grad

from tofea.fea2d import FEA2D_K
from tofea.topopt_helpers import simp_parametrization

max_its = 100
volfrac = 0.5
sigma = 0.5
shape = (160, 80)
nelx, nely = shape
emin, emax = 1e-6, 1

dofs = np.arange(2 * (nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1, 2)
fixed = np.zeros_like(dofs, dtype=bool)
load = np.zeros_like(dofs)

fixed[0, :, :] = 1
load[-1, -1, 1] = 1

fem = FEA2D_K(fixed)
parametrization = simp_parametrization(shape, sigma, emin, emax)
x0 = np.full(shape, volfrac)


def objective(x):
    x = parametrization(x)
    xp = np.pad(x, (0, 1), mode="constant", constant_values=emin)
    xp = (xp - emin) / (emax - emin)
    body = 1e-3 * np.stack([np.zeros_like(xp), xp], axis=-1)
    x = fem(x, load + body)
    return x


def volume(x):
    x = parametrization(x)
    x = (x - emin) / (emax - emin)
    return np.mean(x)


plt.ion()
fig, ax = plt.subplots(1, 1)
im = ax.imshow(parametrization(x0).T, cmap="gray_r", vmin=emin, vmax=emax)
fig.tight_layout()


def volume_constraint(x, gd):
    v, g = value_and_grad(volume)(x)
    if gd.size > 0:
        gd[:] = g
    return v - volfrac


def nlopt_obj(x, gd):
    c, dc = value_and_grad(objective)(x)

    if gd.size > 0:
        gd[:] = dc.ravel()

    im.set_data(parametrization(x).T)
    plt.pause(0.01)

    return c


opt = nlopt.opt(nlopt.LD_CCSAQ, x0.size)
opt.add_inequality_constraint(volume_constraint, 1e-3)
opt.set_min_objective(nlopt_obj)
opt.set_lower_bounds(0)
opt.set_upper_bounds(1)
opt.set_maxeval(max_its)
opt.optimize(x0.ravel())

plt.show(block=True)
