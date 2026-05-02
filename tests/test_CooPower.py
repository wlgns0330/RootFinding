import numpy as np
import pytest

from yroots.polynomial import CooPower, CooCheb

# ---------------------------------------------------------------------
# CooPower evaluation tests
# ---------------------------------------------------------------------

def test_coopower_evaluate_1d_single_point():
    # p(x) = 3 + 2x + 5x^2
    p = CooPower(np.array([3.0, 2.0, 5.0]))

    value = p(2.0)

    expected = 3.0 + 2.0 * 2.0 + 5.0 * 2.0**2
    assert np.allclose(value, expected)

def test_coopower_evaluate_1d_multiple_points():
    # p(x) = 1 - x + 2x^3
    p = CooPower(np.array([1.0, -1.0, 0.0, 2.0]))

    xs = np.array([0.0, 1.0, 2.0])
    values = p(xs)

    expected = 1.0 - xs + 2.0 * xs**3
    assert values.shape == (3,)
    assert np.allclose(values, expected)

def test_coopower_evaluate_2d_single_point_from_coords_data():
    # p(x, y) = 3 + 4xy + 5x^2
    coords = np.array([
        [0, 1, 2],  # powers of x
        [0, 1, 0],  # powers of y
    ])
    data = np.array([3.0, 4.0, 5.0])

    p = CooPower((coords, data), shape=(3, 2))

    value = p(np.array([2.0, 10.0]))

    expected = 3.0 + 4.0 * 2.0 * 10.0 + 5.0 * 2.0**2
    assert np.allclose(value, expected)

def test_coopower_evaluate_2d_multiple_points():
    # p(x, y) = 3 + 4xy + 5x^2
    coords = np.array([
        [0, 1, 2],
        [0, 1, 0],
    ])
    data = np.array([3.0, 4.0, 5.0])

    p = CooPower((coords, data), shape=(3, 2))

    points = np.array([
        [2.0, 10.0],
        [1.0, 5.0],
        [0.0, 3.0],
    ])

    values = p(points)

    x = points[:, 0]
    y = points[:, 1]
    expected = 3.0 + 4.0 * x * y + 5.0 * x**2

    assert values.shape == (3,)
    assert np.allclose(values, expected)

def test_coopower_evaluate_3d_sparse_polynomial():
    # p(x, y, z) = 2xyz + 7z^2
    coords = np.array([
        [1, 0],  # powers of x
        [1, 0],  # powers of y
        [1, 2],  # powers of z
    ])
    data = np.array([2.0, 7.0])

    p = CooPower((coords, data), shape=(2, 2, 3))

    points = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 0.5, 4.0],
    ])

    values = p(points)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    expected = 2.0 * x * y * z + 7.0 * z**2

    assert np.allclose(values, expected)

def test_coopower_zero_polynomial_returns_zero():
    coords = np.empty((2, 0), dtype=int)
    data = np.array([], dtype=float)

    p = CooPower((coords, data), shape=(3, 3))

    points = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    values = p(points)

    assert values.shape == (2,)
    assert np.allclose(values, np.zeros(2))

def test_coopower_zero_polynomial_single_point_returns_scalar_zero():
    coords = np.empty((2, 0), dtype=int)
    data = np.array([], dtype=float)

    p = CooPower((coords, data), shape=(3, 3))

    value = p(np.array([1.0, 2.0]))

    assert np.isscalar(value)
    assert np.allclose(value, 0.0)

def test_coopower_dimension_mismatch_raises():
    p = CooPower(np.zeros((3, 3)))

    with pytest.raises(ValueError):
        p(np.array([1.0, 2.0, 3.0]))


# Unit Tests for power_to_cheb_1d
def terms_to_dense(terms, size=None):
    if size is None:
        size = max(degree for degree, _ in terms) + 1

    coeffs = np.zeros(size, dtype=float)

    for degree, coeff in terms:
        coeffs[degree] = coeff

    return coeffs


def test_power_to_cheb_1d_degree_0():
    terms = CooPower._power_to_cheb_1d(0)

    assert terms == [(0, 1.0)]


def test_power_to_cheb_1d_degree_1():
    terms = CooPower._power_to_cheb_1d(1)

    expected = np.array([0.0, 1.0])
    actual = terms_to_dense(terms, size=2)

    assert np.allclose(actual, expected)


def test_power_to_cheb_1d_degree_2():
    # x^2 = 0.5*T_0 + 0.5*T_2
    terms = CooPower._power_to_cheb_1d(2)

    expected = np.array([0.5, 0.0, 0.5])
    actual = terms_to_dense(terms, size=3)

    assert np.allclose(actual, expected)


def test_power_to_cheb_1d_degree_3():
    # x^3 = 0.75*T_1 + 0.25*T_3
    terms = CooPower._power_to_cheb_1d(3)

    expected = np.array([0.0, 0.75, 0.0, 0.25])
    actual = terms_to_dense(terms, size=4)

    assert np.allclose(actual, expected)


def test_power_to_cheb_1d_degree_4():
    # x^4 = 3/8*T_0 + 1/2*T_2 + 1/8*T_4
    terms = CooPower._power_to_cheb_1d(4)

    expected = np.array([3.0 / 8.0, 0.0, 1.0 / 2.0, 0.0, 1.0 / 8.0])
    actual = terms_to_dense(terms, size=5)

    assert np.allclose(actual, expected)


def test_power_to_cheb_1d_degree_5():
    # x^5 = 10/16*T_1 + 5/16*T_3 + 1/16*T_5
    terms = CooPower._power_to_cheb_1d(5)

    expected = np.array([
        0.0,
        10.0 / 16.0,
        0.0,
        5.0 / 16.0,
        0.0,
        1.0 / 16.0,
    ])
    actual = terms_to_dense(terms, size=6)
    assert np.allclose(actual, expected)

def test_power_to_cheb_1d_negative_degree_raises():
    with pytest.raises(ValueError):
        CooPower._power_to_cheb_1d(-1)

def test_power_to_cheb_1d_dtype():
    terms = CooPower._power_to_cheb_1d(2, dtype=np.float32)
    for _, coeff in terms:
        assert isinstance(coeff, np.float32)


# Unit tests for 1D to_cheb
def assert_coocheb_dense_equal(poly, expected):
    assert isinstance(poly, CooCheb)
    assert poly.shape == expected.shape
    assert np.allclose(poly.coeff.todense(), expected)

def test_to_cheb_returns_coocheb():
    # p(x) = x
    p = CooPower(np.array([0.0, 1.0]))

    q = p.to_cheb()

    assert isinstance(q, CooCheb)
    assert q.basis == "cheb"


def test_to_cheb_constant_1d():
    # p(x) = 3
    p = CooPower(np.array([3.0]))

    q = p.to_cheb()

    expected = np.array([3.0])
    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_linear_1d():
    # p(x) = 2x
    # x = T_1
    p = CooPower(np.array([0.0, 2.0]))

    q = p.to_cheb()

    expected = np.array([0.0, 2.0])
    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_quadratic_1d():
    # p(x) = x^2
    # x^2 = 0.5*T_0 + 0.5*T_2
    p = CooPower(np.array([0.0, 0.0, 1.0]))

    q = p.to_cheb()

    expected = np.array([0.5, 0.0, 0.5])
    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_cubic_1d():
    # p(x) = x^3
    # x^3 = 0.75*T_1 + 0.25*T_3
    p = CooPower(np.array([0.0, 0.0, 0.0, 1.0]))

    q = p.to_cheb()

    expected = np.array([0.0, 0.75, 0.0, 0.25])
    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_mixed_1d_polynomial():
    # p(x) = 3 + 2x + 5x^2
    #      = 3*T_0 + 2*T_1 + 5*(0.5*T_0 + 0.5*T_2)
    #      = 5.5*T_0 + 2*T_1 + 2.5*T_2
    p = CooPower(np.array([3.0, 2.0, 5.0]))

    q = p.to_cheb()

    expected = np.array([5.5, 2.0, 2.5])
    assert_coocheb_dense_equal(q, expected)


# Unit Tests for Nd to_cheb


def test_to_cheb_2d_single_term_x2_y():
    # p(x, y) = x^2 * y
    #
    # x^2 = 0.5*T_0(x) + 0.5*T_2(x)
    # y   = T_1(y)
    #
    # p = 0.5*T_0(x)T_1(y) + 0.5*T_2(x)T_1(y)

    coords = np.array([[2], [1]])
    data = np.array([1.0])

    p = CooPower((coords, data), shape=(3, 2))
    q = p.to_cheb()

    expected = np.zeros((3, 2))
    expected[0, 1] = 0.5
    expected[2, 1] = 0.5

    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_2d_single_term_x2_y3():
    # p(x, y) = x^2 * y^3
    #
    # x^2 = 0.5*T_0(x) + 0.5*T_2(x)
    # y^3 = 0.75*T_1(y) + 0.25*T_3(y)

    coords = np.array([[2], [3]])
    data = np.array([1.0])

    p = CooPower((coords, data), shape=(3, 4))
    q = p.to_cheb()

    expected = np.zeros((3, 4))
    expected[0, 1] = 0.5 * 0.75
    expected[0, 3] = 0.5 * 0.25
    expected[2, 1] = 0.5 * 0.75
    expected[2, 3] = 0.5 * 0.25

    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_2d_with_coefficient():
    # p(x, y) = 8*x^2*y
    coords = np.array([[2], [1]])
    data = np.array([8.0])

    p = CooPower((coords, data), shape=(3, 2))
    q = p.to_cheb()

    expected = np.zeros((3, 2))
    expected[0, 1] = 4.0
    expected[2, 1] = 4.0

    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_2d_multiple_terms_accumulate():
    # p(x, y) = x^2 + y^2
    #
    # x^2 = 0.5*T_0(x)T_0(y) + 0.5*T_2(x)T_0(y)
    # y^2 = 0.5*T_0(x)T_0(y) + 0.5*T_0(x)T_2(y)
    #
    # constant term accumulates to 1.0

    coords = np.array([
        [2, 0],
        [0, 2],
    ])
    data = np.array([1.0, 1.0])

    p = CooPower((coords, data), shape=(3, 3))

    q = p.to_cheb()

    expected = np.zeros((3, 3))
    expected[0, 0] = 1.0
    expected[2, 0] = 0.5
    expected[0, 2] = 0.5

    assert_coocheb_dense_equal(q, expected)


def test_to_cheb_3d_single_term():
    # p(x, y, z) = x^2 * y * z^3
    coords = np.array([
        [2],
        [1],
        [3],
    ])
    data = np.array([2.0])

    p = CooPower((coords, data), shape=(3, 2, 4))

    q = p.to_cheb()

    expected = np.zeros((3, 2, 4))

    # x^2 contributes 0.5*T0 + 0.5*T2
    # y contributes T1
    # z^3 contributes 0.75*T1 + 0.25*T3
    # coefficient is 2.0
    expected[0, 1, 1] = 2.0 * 0.5 * 1.0 * 0.75
    expected[0, 1, 3] = 2.0 * 0.5 * 1.0 * 0.25
    expected[2, 1, 1] = 2.0 * 0.5 * 1.0 * 0.75
    expected[2, 1, 3] = 2.0 * 0.5 * 1.0 * 0.25

    assert_coocheb_dense_equal(q, expected)