# TOFEA

[![tests](https://github.com/yaugenst/tofea/actions/workflows/run_tests.yml/badge.svg)](https://github.com/yaugenst/tofea/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/yaugenst/tofea/graph/badge.svg?token=5Z2SYQ3CPM)](https://codecov.io/gh/yaugenst/tofea)

## Project Overview

TOFEA is a lightweight 2D finite element library for topology optimization.  It can solve heat conduction and small deformation elasticity problems and integrates with [autograd](https://github.com/HIPS/autograd) for automatic differentiation.

## Installation

### Prerequisites
- Python 3.10+

### Steps
1. Clone this repository.
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

The following example is a minimal heat conduction optimization based on
`examples/heat_2d.py`.

```python
import autograd.numpy as np
import matplotlib.pyplot as plt
import nlopt
from autograd import value_and_grad
from tofea.boundary_conditions import BoundaryConditions
from tofea.fea2d import FEA2D_T

shape = (4, 4)
bc = BoundaryConditions(shape)
bc.fix_edge("top")
bc.apply_uniform_load_on_edge("bottom", 1.0)

fem = FEA2D_T(bc.fixed)
x0 = np.full(shape, 0.5)

@value_and_grad
def objective(x):
    t = fem.temperature(x, bc.load)
    return np.mean(t)

def nlopt_obj(x, grad):
    val, g = objective(x.reshape(shape))
    if grad.size > 0:
        grad[:] = g.ravel()
    return val

opt = nlopt.opt(nlopt.LD_MMA, x0.size)
opt.set_lower_bounds(0)
opt.set_upper_bounds(1)
opt.set_min_objective(nlopt_obj)
opt.set_maxeval(5)
x_final = opt.optimize(x0.ravel()).reshape(shape)

plt.imshow(x_final.T, cmap="gray_r")
plt.show()
```

This script defines the problem, runs `nlopt` for a few iterations and displays
the optimized density field.

## Documentation

The full API reference is built with [MkDocs](https://www.mkdocs.org/) and
[mkdocstrings](https://mkdocstrings.github.io/). Install the documentation
extras and run `mkdocs serve` to preview the site locally:

```bash
pip install -e .[docs]
mkdocs serve
```
