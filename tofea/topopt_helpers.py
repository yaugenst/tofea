import autograd.numpy as np
import scipy.ndimage
from autograd.extend import defvjp, primitive

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)


def simp_projection(x, vmin, vmax, penalty=3.0):
    return vmin + x**penalty * (vmin - vmax)


def simp_parametrization(shape, sigma, vmin, vmax, penalty=3.0):
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = simp_projection(x, vmin, vmax, penalty)
        return x

    return _parametrization
