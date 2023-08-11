import autograd.numpy as np
import scipy.ndimage
from autograd.extend import defvjp, primitive

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)


def sigmoid_projection(x, a, b=0.5):
    num = np.tanh(a * b) + np.tanh(a * (x - b))
    denom = np.tanh(a * b) + np.tanh(a * (1 - b))
    return num / denom


def sigmoid_parametrization(shape, sigma, vmin, vmax, alpha=20, beta=0.5, flat=False):
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = sigmoid_projection(x, alpha, beta)
        x = vmin + (vmax - vmin) * x
        return x.ravel() if flat else x

    return _parametrization


def simp_projection(x, vmin, vmax, penalty=3.0):
    return vmin + x**penalty * (vmax - vmin)


def simp_parametrization(shape, sigma, vmin, vmax, penalty=3.0, flat=False):
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = simp_projection(x, vmin, vmax, penalty)
        return x.ravel() if flat else x

    return _parametrization


def gray_indicator(x):
    return np.mean(4 * x * (1 - x))
