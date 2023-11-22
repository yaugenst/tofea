#!/usr/bin/env python3

from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from jax.scipy.signal import convolve

from tofea.fea2d import FEA2D_T


def simp_parametrization(shape, ks, vmin, vmax, penalty=3.0):
    xy = jnp.linspace(-1, 1, ks)
    xx, yy = jnp.meshgrid(xy, xy)
    k = np.sqrt(2) - jnp.sqrt(xx**2 + yy**2)
    k /= jnp.sum(k)

    @jax.jit
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = jnp.pad(x, ks // 2, mode="edge")
        x = convolve(x, k, mode="valid")
        x = vmin + (vmax - vmin) * x**penalty
        return x

    return _parametrization


def main():
    max_its = 100
    volfrac = 0.5
    kernel_size = 5
    shape = (100, 100)
    nelx, nely = shape
    cmin, cmax = 1e-4, 1

    fixed = np.zeros((nelx + 1, nely + 1), dtype="?")
    load = np.zeros_like(fixed)

    fixed[shape[0] // 2 - 5 : shape[0] // 2 + 5, -1] = 1
    load[:, 0] = 1
    load[(0, -1), :] = 1

    fem = FEA2D_T(fixed)
    parametrization = simp_parametrization(shape, kernel_size, cmin, cmax)
    x0 = jnp.full(shape, volfrac)

    plt.ion()
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    im = ax.imshow(parametrization(x0).T, cmap="gray_r", vmin=cmin, vmax=cmax)

    @jax.value_and_grad
    def objective(x):
        x = parametrization(x)
        t = fem.temperature(x, load)
        return jnp.mean(t)

    @jax.value_and_grad
    def volume(x):
        return jnp.mean(x)

    def volume_constraint(x, gd):
        v, g = volume(x)
        if gd.size > 0:
            gd[:] = g.ravel()
        return v.item() - volfrac

    def nlopt_obj(x, gd):
        v, g = objective(x)

        if gd.size > 0:
            gd[:] = g.ravel()

        im.set_data(parametrization(x).T)
        plt.pause(0.01)

        return v.item()

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.add_inequality_constraint(volume_constraint)
    opt.set_min_objective(nlopt_obj)
    opt.set_lower_bounds(0)
    opt.set_upper_bounds(1)
    opt.set_maxeval(max_its)
    t0 = perf_counter()
    opt.optimize(x0.ravel())
    print(perf_counter() - t0)

    plt.show(block=True)


if __name__ == "__main__":
    main()
