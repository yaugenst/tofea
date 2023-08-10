# TOFEA

Simple [autograd](https://github.com/HIPS/autograd)-differentiable finite element analysis for heat conductivity and compliance problems.

## Installation

The package is published on [PyPi](https://pypi.org/), so a simple `pip install tofea` will work.
For development purposes it is recommended to clone this repository and install the package locally instead, i.e. `git clone git@github.com:yaugenst/tofea && pip install -e ./tofea`.

## Examples

The package contains examples of topology optimization for 2D and 3D heat and compliance problems, check them out in the [examples directory](./examples)!

To run the examples, there are some additional dependencies for optimization and plotting, so install using `pip install tofea[examples]`.

## Disclaimer

The package is pretty bare-bones and waiting for a big refactor, which I have not gotten around to.
You are welcome to try everything out as-is but expect the interface to change dramatically in the near future.
