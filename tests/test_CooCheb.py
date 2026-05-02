import numpy as np
import pytest

from yroots.polynomial import CooCheb

def T(k, x):
    """
    Evaluate Chebyshev polynomial T_k(x) using recurrence.
    """
    x = np.asarray(x)

    if k == 0:
        return np.ones_like(x, dtype=np.result_type(x, float))

    if k == 1:
        return x

    t0 = np.ones_like(x, dtype=np.result_type(x, float))
    t1 = x

    for _ in range(2, k + 1):
        t0, t1 = t1, 2 * x * t1 - t0

    return t1

def test_coocheb_evaluate_1d_single_point_degree_0():
    # p(x) = 3*T_0(x) = 3
    p = CooCheb(np.array([3.0]))

    value = p(0.25)

    assert np.isscalar(value)
    assert np.allclose(value, 3.0)


def test_coocheb_evaluate_1d_single_point_degree_1():
    # p(x) = 3*T_0(x) + 2*T_1(x) = 3 + 2x
    p = CooCheb(np.array([3.0, 2.0]))

    x = 0.5
    value = p(x)

    expected = 3.0 + 2.0 * x

    assert np.isscalar(value)
    assert np.allclose(value, expected)


def test_coocheb_evaluate_1d_single_point_degree_2():
    # p(x) = 3*T_0(x) + 2*T_1(x) + 5*T_2(x)
    # T_2(x) = 2x^2 - 1
    p = CooCheb(np.array([3.0, 2.0, 5.0]))

    x = 0.5
    value = p(x)

    expected = 3.0 + 2.0 * T(1, x) + 5.0 * T(2, x)

    assert np.isscalar(value)
    assert np.allclose(value, expected)


def test_coocheb_evaluate_1d_multiple_points():
    # p(x) = 1 - 2*T_1(x) + 4*T_3(x)
    p = CooCheb(np.array([1.0, -2.0, 0.0, 4.0]))

    xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    values = p(xs)

    expected = 1.0 - 2.0 * T(1, xs) + 4.0 * T(3, xs)

    assert values.shape == xs.shape
    assert np.allclose(values, expected)


def test_coocheb_evaluate_2d_single_point_from_coords_data():
    # p(x, y) = 3*T_0(x)T_0(y) + 4*T_1(x)T_1(y) + 5*T_2(x)T_0(y)
    coords = np.array([
        [0, 1, 2],  # degree in x
        [0, 1, 0],  # degree in y
    ])
    data = np.array([3.0, 4.0, 5.0])

    p = CooCheb((coords, data), shape=(3, 2))

    x = 0.25
    y = -0.5
    value = p(np.array([x, y]))

    expected = (
        3.0 * T(0, x) * T(0, y)
        + 4.0 * T(1, x) * T(1, y)
        + 5.0 * T(2, x) * T(0, y)
    )

    assert np.isscalar(value)
    assert np.allclose(value, expected)


def test_coocheb_evaluate_2d_single_point_row_vector_returns_scalar():
    # Same polynomial as above.
    coords = np.array([
        [0, 1, 2],
        [0, 1, 0],
    ])
    data = np.array([3.0, 4.0, 5.0])

    p = CooCheb((coords, data), shape=(3, 2))

    x = 0.25
    y = -0.5
    value = p(np.array([[x, y]]))

    expected = (
        3.0 * T(0, x) * T(0, y)
        + 4.0 * T(1, x) * T(1, y)
        + 5.0 * T(2, x) * T(0, y)
    )

    assert np.isscalar(value)
    assert np.allclose(value, expected)


def test_coocheb_evaluate_2d_multiple_points():
    # p(x, y) = 3 + 4*T_1(x)T_1(y) + 5*T_2(x)
    coords = np.array([
        [0, 1, 2],
        [0, 1, 0],
    ])
    data = np.array([3.0, 4.0, 5.0])

    p = CooCheb((coords, data), shape=(3, 2))

    points = np.array([
        [0.25, -0.5],
        [0.0, 0.75],
        [1.0, -1.0],
    ])

    values = p(points)

    x = points[:, 0]
    y = points[:, 1]

    expected = (
        3.0 * T(0, x) * T(0, y)
        + 4.0 * T(1, x) * T(1, y)
        + 5.0 * T(2, x) * T(0, y)
    )

    assert values.shape == (3,)
    assert np.allclose(values, expected)


def test_coocheb_evaluate_3d_sparse_polynomial():
    # p(x, y, z) = 2*T_1(x)T_1(y)T_1(z) + 7*T_0(x)T_0(y)T_2(z)
    coords = np.array([
        [1, 0],  # degree in x
        [1, 0],  # degree in y
        [1, 2],  # degree in z
    ])
    data = np.array([2.0, 7.0])

    p = CooCheb((coords, data), shape=(2, 2, 3))

    points = np.array([
        [0.25, -0.5, 0.75],
        [1.0, 0.0, -1.0],
    ])

    values = p(points)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    expected = (
        2.0 * T(1, x) * T(1, y) * T(1, z)
        + 7.0 * T(0, x) * T(0, y) * T(2, z)
    )

    assert values.shape == (2,)
    assert np.allclose(values, expected)


def test_coocheb_zero_polynomial_multiple_points_returns_zero_array():
    coords = np.empty((2, 0), dtype=int)
    data = np.array([], dtype=float)

    p = CooCheb((coords, data), shape=(3, 3))

    points = np.array([
        [0.0, 0.0],
        [0.5, -0.5],
        [1.0, 1.0],
    ])

    values = p(points)

    assert values.shape == (3,)
    assert np.allclose(values, np.zeros(3))


def test_coocheb_zero_polynomial_single_point_returns_scalar_zero():
    coords = np.empty((2, 0), dtype=int)
    data = np.array([], dtype=float)

    p = CooCheb((coords, data), shape=(3, 3))

    value = p(np.array([0.5, -0.5]))

    assert np.isscalar(value)
    assert np.allclose(value, 0.0)


def test_coocheb_dimension_mismatch_raises():
    p = CooCheb(np.zeros((3, 3)))

    with pytest.raises(ValueError):
        p(np.array([1.0, 2.0, 3.0]))