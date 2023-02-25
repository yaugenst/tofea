#!/usr/bin/env python3

import os

import autograd.numpy as np
import matplotlib.pyplot as plt
import nlopt
import scipy.ndimage
from autograd import tensor_jacobian_product, value_and_grad
from autograd.extend import defvjp, primitive
from tofem import FEM2D_K, FEM2D_T
from matplotlib.colors import CenteredNorm

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_WARNINGS"] = "0"

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)


def create_parametrization(shape, sigma, e_min=1e-4, e_max=1, p=3.0):
    def parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = e_min + x**p * (e_max - e_min)
        return x

    return parametrization


def run_mech():
    max_its = 100
    volfrac = 0.4
    sigma = 1.0
    shape = (160, 80)
    nelx, nely = shape
    emin = 1e-4
    emax = 1

    dofs = np.arange(2 * (nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1, 2)
    fixed = np.zeros_like(dofs, dtype=bool)
    load = np.zeros_like(dofs)

    fixed[0, :, :] = 1
    load[-1, -1, 1] = 1

    x0 = np.full(shape, volfrac)
    fem = FEM2D_K(shape, dofs, fixed, load, solver="direct")
    parametrization = create_parametrization(shape, sigma)

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
            design.T * np.reshape(fem.displacement, load.shape)[1:, 1:, 0].T,
            cmap="RdBu",
            norm=CenteredNorm(),
        )
        ax[2].cla()
        ax[2].imshow(
            design.T * np.reshape(fem.displacement, load.shape)[1:, 1:, 1].T,
            cmap="RdBu",
            norm=CenteredNorm(),
        )
        ax[3].cla()
        ax[3].imshow(-fem._QeKQe.T, cmap="magma")
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


def run_temp():
    max_its = 100
    volfrac = 0.3
    sigma = 1.0
    shape = (100, 100)
    nelx, nely = shape
    cmin, cmax = 1.0, 1e6

    dofs = np.arange((nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1)
    fixed = np.zeros_like(dofs, dtype=bool)
    load = np.zeros_like(dofs)

    fixed[40:60, -1] = 1

    load[:, 0] = 1

    x0 = np.full(shape, volfrac)

    fem = FEM2D_T(shape, dofs, fixed, load, solver="direct")
    parametrization = create_parametrization(shape, sigma, cmin, cmax)

    plt.ion()
    fig, ax = plt.subplots(2, 1)
    im1 = ax[0].imshow(parametrization(x0).T, cmap="gray_r", vmin=cmin, vmax=cmax)
    ax[1].imshow(np.zeros_like(x0), cmap="magma", vmin=0, vmax=1)
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
        ax[1].imshow(-fem._QeKQe.T, cmap="magma")
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


if __name__ == "__main__":
    run_mech()
    run_temp()
