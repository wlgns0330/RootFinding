# tests/test_solve_coo.py

import numpy as np
import pytest
from sparse import COO

import yroots as yr

# Adjust these imports to your actual module names.
# from yroots.CooPolynomial import CooPower, CooCheb
from yroots.polynomial import MultiPower, MultiCheb, CooPower, CooCheb


def assert_roots_match(actual, expected, atol=1e-8):
    """
    Compare unordered root arrays.
    Assumes both arrays have shape (n_roots, dim).
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    assert actual.ndim == 2
    assert expected.ndim == 2
    assert actual.shape == expected.shape

    used = np.zeros(len(expected), dtype=bool)

    for root in actual:
        dists = np.linalg.norm(expected - root, axis=1)
        dists[used] = np.inf
        j = np.argmin(dists)

        assert dists[j] < atol, f"Root {root} did not match expected roots {expected}"
        used[j] = True


def test_solve_coo_power_matches_dense_power_2d_linear_system():
    """
    System:
        f(x, y) = x - 0.25
        g(x, y) = y + 0.5

    Root:
        (0.25, -0.5)
    """

    f = np.zeros((2, 1), dtype=float)
    g = np.zeros((1, 2), dtype=float)

    # Power basis:
    # f = -0.25 + x
    # g =  0.50 + y
    f[0, 0] = -0.25
    f[1, 0] = 1.0

    g[0, 0] = 0.5
    g[0, 1] = 1.0

    dense_polys = [
        MultiPower(f),
        MultiPower(g),
    ]

    coo_polys = [
        CooPower(f),
        CooPower(g),
    ]

    dense_roots = yr.solve(dense_polys)
    coo_roots = yr.solve(coo_polys)

    expected = np.array([[0.25, -0.5]])

    assert_roots_match(dense_roots, expected)
    assert_roots_match(coo_roots, dense_roots)


def test_solve_coo_cheb_matches_dense_cheb_2d_linear_system():
    """
    Chebyshev system:
        f(x, y) = T1(x) - 0.25
        g(x, y) = T1(y) + 0.5

    Since T1(t) = t, root is:
        (0.25, -0.5)
    """

    f = np.zeros((2, 1), dtype=float)
    g = np.zeros((1, 2), dtype=float)

    # Chebyshev basis:
    # f = -0.25*T0 + 1*T1(x)
    # g =  0.50*T0 + 1*T1(y)
    f[0, 0] = -0.25
    f[1, 0] = 1.0

    g[0, 0] = 0.5
    g[0, 1] = 1.0

    dense_polys = [
        MultiCheb(f),
        MultiCheb(g),
    ]

    coo_polys = [
        CooPower(f),
        CooPower(g),
    ]

    dense_roots = yr.solve(dense_polys)
    coo_roots = yr.solve(coo_polys)

    expected = np.array([[0.25, -0.5]])

    assert_roots_match(dense_roots, expected)
    assert_roots_match(coo_roots, dense_roots)


def test_solve_coo_cheb_matches_dense_cheb_2d_quadratic_system():
    """
    System:
        f(x, y) = x^2 - 0.25
        g(x, y) = y - 0.25

    In Chebyshev basis:
        x^2 = (T2(x) + T0) / 2

    So:
        x^2 - 0.25 = 0.25*T0 + 0.5*T2(x)

    Roots in [-1, 1]^2:
        (-0.5, 0.25)
        ( 0.5, 0.25)
    """

    f = np.zeros((3, 1), dtype=float)
    g = np.zeros((1, 2), dtype=float)

    # f = x^2 - 0.25 = 0.25*T0 + 0.5*T2(x)
    f[0, 0] = 0.25
    f[2, 0] = 0.5

    # g = y - 0.25 = -0.25*T0 + T1(y)
    g[0, 0] = -0.25
    g[0, 1] = 1.0

    dense_polys = [
        MultiCheb(f),
        MultiCheb(g),
    ]

    coo_polys = [
        CooCheb(f),
        CooCheb(g),
    ]

    dense_roots = yr.solve(dense_polys)
    coo_roots = yr.solve(coo_polys)

    expected = np.array([
        [-0.5, 0.25],
        [0.5, 0.25],
    ])

    assert_roots_match(dense_roots, expected)
    assert_roots_match(coo_roots, dense_roots)


def test_solve_coo_power_matches_dense_power_2d_quadratic_system():
    """
    Power basis version of:
        f(x, y) = x^2 - 0.25
        g(x, y) = y - 0.25

    Roots:
        (-0.5, 0.25)
        ( 0.5, 0.25)
    """

    f = np.zeros((3, 1), dtype=float)
    g = np.zeros((1, 2), dtype=float)

    # f = -0.25 + x^2
    f[0, 0] = -0.25
    f[2, 0] = 1.0

    # g = -0.25 + y
    g[0, 0] = -0.25
    g[0, 1] = 1.0

    dense_polys = [
        MultiPower(f),
        MultiPower(g),
    ]

    coo_polys = [
        CooPower(f),
        CooPower(g),
    ]

    dense_roots = yr.solve(dense_polys)
    coo_roots = yr.solve(coo_polys)

    expected = np.array([
        [-0.5, 0.25],
        [0.5, 0.25],
    ])

    assert_roots_match(dense_roots, expected)
    assert_roots_match(coo_roots, dense_roots)


def test_solve_coo_cheb_matches_dense_cheb_3d_linear_system():
    """
    3D Chebyshev linear system:
        x - 0.2 = 0
        y + 0.3 = 0
        z - 0.4 = 0

    Root:
        (0.2, -0.3, 0.4)
    """

    f = np.zeros((2, 1, 1), dtype=float)
    g = np.zeros((1, 2, 1), dtype=float)
    h = np.zeros((1, 1, 2), dtype=float)

    f[0, 0, 0] = -0.2
    f[1, 0, 0] = 1.0

    g[0, 0, 0] = 0.3
    g[0, 1, 0] = 1.0

    h[0, 0, 0] = -0.4
    h[0, 0, 1] = 1.0

    dense_polys = [
        MultiCheb(f),
        MultiCheb(g),
        MultiCheb(h),
    ]

    coo_polys = [
        CooPower(f),
        CooPower(g),
        CooPower(h),
    ]

    dense_roots = yr.solve(dense_polys)
    coo_roots = yr.solve(coo_polys)

    expected = np.array([[0.2, -0.3, 0.4]])

    assert_roots_match(dense_roots, expected)
    assert_roots_match(coo_roots, dense_roots)


def test_solve_coo_power_matches_dense_power_3d_linear_system():
    """
    3D power-basis linear system:
        x - 0.2 = 0
        y + 0.3 = 0
        z - 0.4 = 0

    Root:
        (0.2, -0.3, 0.4)
    """

    f = np.zeros((2, 1, 1), dtype=float)
    g = np.zeros((1, 2, 1), dtype=float)
    h = np.zeros((1, 1, 2), dtype=float)

    f[0, 0, 0] = -0.2
    f[1, 0, 0] = 1.0

    g[0, 0, 0] = 0.3
    g[0, 1, 0] = 1.0

    h[0, 0, 0] = -0.4
    h[0, 0, 1] = 1.0

    dense_polys = [
        MultiPower(f),
        MultiPower(g),
        MultiPower(h),
    ]

    coo_polys = [
        CooPower(f),
        CooPower(g),
        CooPower(h),
    ]

    dense_roots = yr.solve(dense_polys)
    coo_roots = yr.solve(coo_polys)

    expected = np.array([[0.2, -0.3, 0.4]])

    assert_roots_match(dense_roots, expected)
    assert_roots_match(coo_roots, dense_roots)


# test_solve_mixed_dim3_deg5.py

import itertools
import math

POLY_TYPES = [MultiCheb, MultiPower, CooCheb, CooPower]

def assert_root_found(roots, expected, tol=1e-7):
    roots = np.asarray(roots)

    assert roots.ndim == 2
    assert roots.shape[1] == len(expected)

    dists = np.linalg.norm(roots - expected, axis=1)
    assert np.min(dists) < tol, (
        f"Expected root {expected} not found.\n"
        f"Closest distance: {np.min(dists)}\n"
        f"Roots:\n{roots}"
    )


def make_power_coeff_for_dim3_deg5(axis, root, scale=0.05):
    """
    Build a dim-3 degree-5 power-basis coefficient tensor for

        f(x_axis) = (x_axis - root) + scale * (x_axis - root)^5

    This has the desired root at x_axis = root and includes a real degree-5 term.
    Shape is (6, 6, 6), so degree 5 in each dimension is supported.
    """
    coeff = np.zeros((6, 6, 6), dtype=float)

    # Linear part: x - root
    idx0 = [0, 0, 0]
    coeff[tuple(idx0)] += -root

    idx1 = [0, 0, 0]
    idx1[axis] = 1
    coeff[tuple(idx1)] += 1.0

    # Degree-5 part: scale * (x - root)^5
    # (x - r)^5 = sum_{k=0}^5 binom(5,k) x^k (-r)^(5-k)
    for k in range(6):
        idx = [0, 0, 0]
        idx[axis] = k
        coeff[tuple(idx)] += scale * math.comb(5, k) * ((-root) ** (5 - k))

    return coeff


def make_base_power_system():
    """
    System has known root:

        x =  0.2
        y = -0.3
        z =  0.4

    Each polynomial is degree 5 but depends on only one variable.
    """
    expected_root = np.array([0.2, -0.3, 0.4], dtype=float)

    power_coeffs = [
        make_power_coeff_for_dim3_deg5(axis=0, root=expected_root[0]),
        make_power_coeff_for_dim3_deg5(axis=1, root=expected_root[1]),
        make_power_coeff_for_dim3_deg5(axis=2, root=expected_root[2]),
    ]

    return power_coeffs, expected_root


def as_poly(poly_type, power_coeff):
    """
    Convert a dense power-basis coefficient tensor into the requested polynomial type.
    """
    if poly_type is MultiPower:
        return MultiPower(power_coeff.copy())

    if poly_type is MultiCheb:
        cheb_coeff = MultiPower(power_coeff.copy()).to_cheb()
        if isinstance(cheb_coeff, MultiCheb):
            return cheb_coeff
        return MultiCheb(cheb_coeff)

    if poly_type is CooPower:
        return CooPower(power_coeff.copy())

    if poly_type is CooCheb:
        cheb_coeff = CooPower(power_coeff.copy()).to_cheb()
        if isinstance(cheb_coeff, CooCheb):
            return cheb_coeff
        return CooCheb(cheb_coeff)

    raise TypeError(f"Unsupported polynomial type: {poly_type}")


@pytest.mark.parametrize(
    "type_combo",
    list(itertools.combinations(POLY_TYPES, 3)),
    ids=lambda combo: "_".join(t.__name__ for t in combo),
)
def test_solve_dim3_degree5_mixed_combinations_of_three(type_combo):
    """
    Tests every combination of 3 polynomial representations chosen from:

        MultiCheb, MultiPower, CooCheb, CooPower

    Example combinations:
        MultiCheb, MultiPower, CooCheb
        MultiCheb, MultiPower, CooPower
        MultiCheb, CooCheb, CooPower
        MultiPower, CooCheb, CooPower

    The solver should accept the mixed system and recover the known root.
    """
    power_coeffs, expected_root = make_base_power_system()

    funcs = [
        as_poly(poly_type, coeff)
        for poly_type, coeff in zip(type_combo, power_coeffs)
    ]

    roots = yr.solve(funcs)

    assert_root_found(roots, expected_root)


def test_solve_dim3_degree5_dense_baseline():
    """
    Baseline check using only dense MultiPower inputs.
    This helps distinguish a mixed-input failure from a bad test system.
    """
    power_coeffs, expected_root = make_base_power_system()

    funcs = [MultiPower(coeff.copy()) for coeff in power_coeffs]

    roots = yr.solve(funcs)

    assert_root_found(roots, expected_root)


def test_solve_dim3_degree5_coo_baseline():
    """
    Baseline check using only CooPower inputs.
    This helps distinguish mixed-input failures from sparse-only failures.
    """
    power_coeffs, expected_root = make_base_power_system()

    funcs = [CooPower(coeff.copy()) for coeff in power_coeffs]

    roots = yr.solve(funcs)

    assert_root_found(roots, expected_root)