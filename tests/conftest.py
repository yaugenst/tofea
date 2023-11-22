import pytest
from numpy.random import Generator, default_rng


@pytest.fixture(scope="session")
def rng() -> Generator:
    seed = 365235228
    return default_rng(seed)
