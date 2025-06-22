import tofea


def test_public_attributes():
    assert isinstance(tofea.__version__, str)
    assert tofea.DEFAULT_SOLVER == "SuperLU"
    for name in [
        "DEFAULT_SOLVER",
        "__version__",
        "boundary_conditions",
        "elements",
        "fea2d",
        "primitives",
        "solvers",
    ]:
        assert name in tofea.__all__
