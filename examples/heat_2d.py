#!/usr/bin/env python3
"""Heat conduction optimization example using 2D finite elements."""

import autograd.numpy as np
import matplotlib.pyplot as plt
import nlopt
import scipy.ndimage
from autograd import value_and_grad
from autograd.extend import defvjp, primitive

from tofea.boundary_conditions import BoundaryConditions
from tofea.fea2d import FEA2D_T

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),  # noqa: ARG005
)


def simp_parametrization(shape, sigma, vmin, vmax, penalty=3.0):
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = vmin + (vmax - vmin) * x**penalty
        return x

    return _parametrization


def main():
    max_its = 100
    volfrac = 0.5
    sigma = 1.0
    shape = (100, 100)
    cmin, cmax = 1e-4, 1

    bc = BoundaryConditions(shape)
    bc.fix_edge("top")
    bc.apply_uniform_load_on_edge("bottom", 1.0)
    bc.apply_uniform_load_on_edge("left", 1.0)
    bc.apply_uniform_load_on_edge("right", 1.0)

    load = bc.load
    fem = FEA2D_T(bc.fixed)
    parametrization = simp_parametrization(shape, sigma, cmin, cmax)
    x0 = np.full(shape, volfrac)

    plt.ion()
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    im = ax.imshow(parametrization(x0).T, cmap="gray_r", vmin=cmin, vmax=cmax)

    @value_and_grad
    def objective(x):
        x = parametrization(x)
        t = fem.temperature(x, load)
        return np.mean(t)

    @value_and_grad
    def volume(x):
        return np.mean(x)

    def volume_constraint(x, gd):
        v, g = volume(x)
        if gd.size > 0:
            gd[:] = g.ravel()
        return v - volfrac

    def nlopt_obj(x, gd):
        v, g = objective(x)

        if gd.size > 0:
            gd[:] = g.ravel()

        im.set_data(parametrization(x).T)
        plt.pause(0.01)

        return v

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.add_inequality_constraint(volume_constraint)
    opt.set_min_objective(nlopt_obj)
    opt.set_lower_bounds(0)
    opt.set_upper_bounds(1)
    opt.set_maxeval(max_its)
    opt.optimize(x0.ravel())

    plt.show(block=True)


if __name__ == "__main__":
    main()
