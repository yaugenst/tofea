import autograd.test_util as agtu

# Relax autograd gradient check tolerances slightly to account for
# small numerical differences across platforms and Python versions.
agtu.TOL = 1e-5
agtu.RTOL = 1e-5
