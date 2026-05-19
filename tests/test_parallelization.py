import numpy as np
import pytest
import sys
import sysconfig

import yroots as yr
from yroots import MultiPower


def sort_roots(roots):
    """
    Sort roots row-wise so parallel results can be compared independent
    of output ordering.
    """
    roots = np.asarray(roots, dtype=float)

    if roots.size == 0:
        return roots.reshape(0, 0)

    return roots[np.lexsort(roots.T[::-1])]

def test_python_is_314t():
    assert sys.version_info[:2] == (3, 14)
    assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1

def assert_same_roots(actual, expected, tol=1e-10):
    actual = sort_roots(actual)
    expected = sort_roots(expected)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=tol, rtol=tol)


def simple_2d_system():
    """
    System:
        f1(x, y) = x^2 - 1/4
        f2(x, y) = y^2 - 1/9

    Roots in [-1, 1]^2:
        x = +/- 1/2
        y = +/- 1/3
    """

    # coeff[i, j] corresponds to x^i y^j in MultiPower
    f1_coeff = np.zeros((3, 1))
    f1_coeff[0, 0] = -0.25
    f1_coeff[2, 0] = 1.0

    f2_coeff = np.zeros((1, 3))
    f2_coeff[0, 0] = -1.0 / 9.0
    f2_coeff[0, 2] = 1.0

    funcs = [
        MultiPower(f1_coeff),
        MultiPower(f2_coeff),
    ]

    a = np.array([-1.0, -1.0])
    b = np.array([1.0, 1.0])

    return funcs, a, b


def test_negative_max_cpu_raises_error():
    funcs, a, b = simple_2d_system()

    with pytest.raises(ValueError):
        yr.solve(
            funcs,
            a,
            b,
            max_cpu=-1,
            parallel_depth=1,
        )


@pytest.mark.parametrize("bad_parallel_depth", [-1, 3, 10])
def test_invalid_parallel_depth_raises_error(bad_parallel_depth):
    funcs, a, b = simple_2d_system()

    with pytest.raises(ValueError):
        yr.solve(
            funcs,
            a,
            b,
            max_cpu=2,
            parallel_depth=bad_parallel_depth,
        )


def test_parallel_depth_0_1_2_give_same_roots():
    funcs, a, b = simple_2d_system()

    roots_serial = yr.solve(
        funcs,
        a,
        b,
        max_cpu=1,
        parallel_depth=0,
    )

    roots_depth_1 = yr.solve(
        funcs,
        a,
        b,
        max_cpu=8,
        parallel_depth=1,
    )

    roots_depth_2 = yr.solve(
        funcs,
        a,
        b,
        max_cpu=8,
        parallel_depth=2,
    )

    assert_same_roots(roots_depth_1, roots_serial)
    assert_same_roots(roots_depth_2, roots_serial)