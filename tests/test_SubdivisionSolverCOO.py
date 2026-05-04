import numpy as np
import pytest
from sparse import COO

# Change this import to match your project structure
from yroots.ChebyshevSubdivisionSolver import TransformChebInPlace1D
from yroots.ChebyshevSubdivisionSolverCOO import build_cheb_transform_matrix, TransformChebInPlace1DCOO, TransformChebInPlace1DCOO_manual, TransformChebInPlace1DCOO_manual2


def test_build_cheb_transform_matrix_n1():
    C = build_cheb_transform_matrix(1, alpha=0.5, beta=0.25)

    expected = np.array([[1.0]])

    assert C.shape == (1, 1)
    np.testing.assert_allclose(C, expected)


def test_build_cheb_transform_matrix_degree_1():
    alpha = 0.5
    beta = 0.25

    C = build_cheb_transform_matrix(2, alpha, beta)

    # T_0(alpha*x + beta) = T_0
    # T_1(alpha*x + beta) = beta*T_0 + alpha*T_1
    expected = np.array([
        [1.0, beta],
        [0.0, alpha],
    ])

    np.testing.assert_allclose(C, expected)


def test_build_cheb_transform_matrix_identity():
    n = 6

    C = build_cheb_transform_matrix(n, alpha=1.0, beta=0.0)

    np.testing.assert_allclose(C, np.eye(n), atol=1e-14)


def test_build_cheb_transform_matrix_shift_only_alpha_zero():
    n = 5
    beta = 0.3

    C = build_cheb_transform_matrix(n, alpha=0.0, beta=beta)

    # If alpha = 0, then T_j(alpha*x + beta) = T_j(beta),
    # which is a constant. Therefore only row 0 should be nonzero.
    assert C.shape == (n, n)
    np.testing.assert_allclose(C[1:, :], 0.0, atol=1e-14)

    # Check the constants T_j(beta).
    expected_row0 = np.array([
        np.cos(j * np.arccos(beta)) for j in range(n)
    ])

    np.testing.assert_allclose(C[0, :], expected_row0, atol=1e-14)


def test_build_cheb_transform_matrix_known_degree_2():
    alpha = 0.5
    beta = 0.25

    C = build_cheb_transform_matrix(3, alpha, beta)

    # T_2(alpha*x + beta)
    # = 2(alpha*x + beta)^2 - 1
    # = 2 alpha^2 x^2 + 4 alpha beta x + 2 beta^2 - 1
    #
    # Use x^2 = (T_2 + T_0) / 2
    #
    # constant term:
    # alpha^2 + 2 beta^2 - 1
    #
    # T_1 term:
    # 4 alpha beta
    #
    # T_2 term:
    # alpha^2
    expected = np.array([
        [1.0, beta, alpha**2 + 2 * beta**2 - 1],
        [0.0, alpha, 4 * alpha * beta],
        [0.0, 0.0, alpha**2],
    ])

    np.testing.assert_allclose(C, expected, atol=1e-14)


def test_build_cheb_transform_matrix_known_degree_3():
    alpha = 0.5
    beta = 0.25

    C = build_cheb_transform_matrix(4, alpha, beta)

    # Expected columns:
    #
    # col 0: T_0(alpha*x + beta)
    # col 1: T_1(alpha*x + beta)
    # col 2: T_2(alpha*x + beta)
    # col 3: T_3(alpha*x + beta)
    #
    # T_3(y) = 4y^3 - 3y, y = alpha*x + beta.
    #
    # Convert powers of x to Chebyshev:
    # x = T_1
    # x^2 = (T_2 + T_0) / 2
    # x^3 = (T_3 + 3T_1) / 4

    expected = np.zeros((4, 4))

    expected[:, 0] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    expected[:, 1] = [
        beta,
        alpha,
        0.0,
        0.0,
    ]

    expected[:, 2] = [
        alpha**2 + 2 * beta**2 - 1,
        4 * alpha * beta,
        alpha**2,
        0.0,
    ]

    expected[:, 3] = [
        beta * (6 * alpha**2 + 4 * beta**2 - 3),
        alpha * (3 * alpha**2 + 12 * beta**2 - 3),
        6 * alpha**2 * beta,
        alpha**3,
    ]

    np.testing.assert_allclose(C, expected, atol=1e-14)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_build_cheb_transform_matrix_matches_original_1d_transform(n):
    alpha = 0.6
    beta = -0.2

    rng = np.random.default_rng(12345)
    coeffs = rng.normal(size=n)

    C = build_cheb_transform_matrix(n, alpha, beta)

    actual = C @ coeffs
    expected = TransformChebInPlace1D_manual2(coeffs, alpha, beta)

    # TransformChebInPlace1D may return transformedCoeffs[:maxRow],
    # so pad it if needed.
    if expected.shape[0] < n:
        padded = np.zeros(n)
        padded[:expected.shape[0]] = expected
        expected = padded

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_transform_cheb_in_place_1d_coo_matches_dense_1d_vector():
    alpha = 0.6
    beta = -0.2

    dense_coeffs = np.array([1.0, 0.0, 2.0, 0.0, -0.5])
    sparse_coeffs = COO.from_numpy(dense_coeffs)

    actual_sparse = TransformChebInPlace1DCOO(sparse_coeffs, alpha, beta)
    actual = actual_sparse.todense()

    expected = TransformChebInPlace1D(dense_coeffs, alpha, beta)

    if expected.shape[0] < dense_coeffs.shape[0]:
        padded = np.zeros(dense_coeffs.shape[0])
        padded[:expected.shape[0]] = expected
        expected = padded

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_transform_cheb_in_place_1d_coo_matches_dense_nd_axis0():
    alpha = 0.6
    beta = -0.2

    dense_coeffs = np.zeros((5, 3, 2))
    dense_coeffs[0, 0, 0] = 1.0
    dense_coeffs[2, 1, 0] = 2.0
    dense_coeffs[4, 2, 1] = -0.5

    sparse_coeffs = COO.from_numpy(dense_coeffs)

    actual_sparse = TransformChebInPlace1DCOO(sparse_coeffs, alpha, beta)
    actual = actual_sparse.todense()

    expected = TransformChebInPlace1D(dense_coeffs, alpha, beta)

    np.testing.assert_allclose(actual, expected, atol=1e-12)

def test_transform_cheb_in_place_1d_coo_manual_matches_dense_nd_axis0():
    alpha = 0.6
    beta = -0.2

    dense_coeffs = np.zeros((5, 3, 2))
    dense_coeffs[0, 0, 0] = 1.0
    dense_coeffs[2, 1, 0] = 2.0
    dense_coeffs[4, 2, 1] = -0.5

    sparse_coeffs = COO.from_numpy(dense_coeffs)
    
    actual_sparse = TransformChebInPlace1DCOO_manual(sparse_coeffs, alpha, beta)
    actual = actual_sparse.todense()

    expected = TransformChebInPlace1D(dense_coeffs, alpha, beta)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_transform_cheb_in_place_1d_coo_manual2_matches_dense_nd_axis0():
    alpha = 0.6
    beta = -0.2

    dense_coeffs = np.zeros((5, 3, 2))
    dense_coeffs[0, 0, 0] = 1.0
    dense_coeffs[2, 1, 0] = 2.0
    dense_coeffs[4, 2, 1] = -0.5

    sparse_coeffs = COO.from_numpy(dense_coeffs)

    actual_sparse = TransformChebInPlace1DCOO_manual2(sparse_coeffs, alpha, beta)
    actual = actual_sparse.todense()

    expected = TransformChebInPlace1D(dense_coeffs, alpha, beta)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (4, 2),
        (5, 3, 2),
    ],
)
def test_transform_cheb_in_place_1d_coo_preserves_shape(shape):
    alpha = 0.5
    beta = 0.1

    dense_coeffs = np.zeros(shape)
    dense_coeffs[0] = 1.0

    sparse_coeffs = COO.from_numpy(dense_coeffs)
    actual = TransformChebInPlace1DCOO_manual2(sparse_coeffs, alpha, beta)

    assert actual.shape == shape