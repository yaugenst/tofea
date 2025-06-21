# TOFEA

[![tests](https://github.com/yaugenst/tofea/actions/workflows/run_tests.yml/badge.svg)](https://github.com/yaugenst/tofea/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/yaugenst/tofea/graph/badge.svg?token=5Z2SYQ3CPM)](https://codecov.io/gh/yaugenst/tofea)

Simple [autograd](https://github.com/HIPS/autograd)-differentiable finite element analysis for heat conductivity and compliance problems.

## Installation

The package is published on [PyPi](https://pypi.org/), so a simple `pip install tofea` will work.
For development purposes it is recommended to clone this repository and install the package locally instead, i.e. `git clone git@github.com:yaugenst/tofea && pip install -e ./tofea`.

## Examples

The package contains examples of topology optimization for 2D and 3D heat and compliance problems, check them out in the [examples directory](./examples)!

To run the examples, there are some additional dependencies for optimization and plotting, so install using `pip install tofea[examples]`.

```bash
python examples/compliance_2d.py
```

This will start an optimization run and display the design evolution in a
window. The heat conduction example can be run in the same way.

## Documentation

The API reference is built with [MkDocs](https://www.mkdocs.org/) and
[mkdocstrings](https://mkdocstrings.github.io/). Install the documentation
extras and run `mkdocs serve` to preview the site locally:

```bash
pip install -e .[docs]
mkdocs serve
```

## Disclaimer

The package is pretty bare-bones and waiting for a big refactor, which I have not gotten around to.
You are welcome to try everything out as-is but expect the interface to change dramatically in the near future.
