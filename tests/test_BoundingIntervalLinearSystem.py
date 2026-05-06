import numpy as np
import pytest
from sparse import COO

# Import from your actual module
from yroots.ChebyshevSubdivisionSolver import BoundingIntervalLinearSystem


def assert_bounding_results_equal(coo_result, dense_result, atol=1e-12, rtol=1e-12):
    coo_interval, coo_changed, coo_should_stop, coo_throwout = coo_result
    dense_interval, dense_changed, dense_should_stop, dense_throwout = dense_result

    np.testing.assert_allclose(
        coo_interval,
        dense_interval,
        atol=atol,
        rtol=rtol,
    )

    assert coo_changed == dense_changed
    assert coo_should_stop == dense_should_stop
    assert coo_throwout == dense_throwout


def test_BoundingIntervalLinearSystem_coo_matches_dense_2d():
    # Two 2D Chebyshev coefficient tensors for a 2-equation system.
    M0 = np.zeros((4, 4), dtype=np.float64)
    M1 = np.zeros((4, 4), dtype=np.float64)

    # Polynomial 0:
    #   0.2 + 1.5*T1(x) - 0.25*T1(y) + small higher terms
    M0[0, 0] = 0.2
    M0[1, 0] = 1.5
    M0[0, 1] = -0.25
    M0[2, 0] = 0.03
    M0[0, 2] = -0.01
    M0[3, 1] = 0.005

    # Polynomial 1:
    #   -0.1 + 0.4*T1(x) + 1.2*T1(y) + small higher terms
    M1[0, 0] = -0.1
    M1[1, 0] = 0.4
    M1[0, 1] = 1.2
    M1[1, 1] = -0.02
    M1[0, 3] = 0.004

    dense_Ms = [M0.copy(), M1.copy()]
    coo_Ms = [COO.from_numpy(M) for M in dense_Ms]

    errors_dense = np.array([1e-8, 2e-8], dtype=np.float64)
    errors_coo = errors_dense.copy()

    dense_result = BoundingIntervalLinearSystem(
        dense_Ms,
        errors_dense,
        finalStep=False,
    )

    coo_result = BoundingIntervalLinearSystem(
        coo_Ms,
        errors_coo,
        finalStep=False,
    )

    assert_bounding_results_equal(coo_result, dense_result)


def test_BoundingIntervalLinearSystem_coo_matches_dense_3d():
    # Three 3D Chebyshev coefficient tensors for a 3-equation system.
    M0 = np.zeros((4, 4, 4), dtype=np.float64)
    M1 = np.zeros((4, 4, 4), dtype=np.float64)
    M2 = np.zeros((4, 4, 4), dtype=np.float64)

    # Make the linear system reasonably well-conditioned.
    M0[0, 0, 0] = 0.1
    M0[1, 0, 0] = 1.3
    M0[0, 1, 0] = 0.2
    M0[0, 0, 1] = -0.1
    M0[2, 0, 0] = 0.01
    M0[0, 3, 0] = -0.004

    M1[0, 0, 0] = -0.2
    M1[1, 0, 0] = -0.15
    M1[0, 1, 0] = 1.1
    M1[0, 0, 1] = 0.25
    M1[1, 1, 0] = 0.02
    M1[0, 0, 3] = 0.003

    M2[0, 0, 0] = 0.05
    M2[1, 0, 0] = 0.1
    M2[0, 1, 0] = -0.2
    M2[0, 0, 1] = 1.4
    M2[0, 2, 0] = -0.015
    M2[3, 0, 0] = 0.002

    dense_Ms = [M0.copy(), M1.copy(), M2.copy()]
    coo_Ms = [COO.from_numpy(M) for M in dense_Ms]

    errors_dense = np.array([1e-8, 2e-8, 3e-8], dtype=np.float64)
    errors_coo = errors_dense.copy()

    dense_result = BoundingIntervalLinearSystem(
        dense_Ms,
        errors_dense,
        finalStep=False,
    )

    coo_result = BoundingIntervalLinearSystem(
        coo_Ms,
        errors_coo,
        finalStep=False,
    )

    assert_bounding_results_equal(coo_result, dense_result)


def test_BoundingIntervalLinearSystem_coo_matches_dense_final_step():
    M0 = np.zeros((4, 4), dtype=np.float64)
    M1 = np.zeros((4, 4), dtype=np.float64)

    M0[0, 0] = 0.15
    M0[1, 0] = 1.0
    M0[0, 1] = 0.2
    M0[2, 0] = 0.01

    M1[0, 0] = -0.25
    M1[1, 0] = -0.3
    M1[0, 1] = 1.2
    M1[0, 2] = -0.02

    dense_Ms = [M0.copy(), M1.copy()]
    coo_Ms = [COO.from_numpy(M) for M in dense_Ms]

    # finalStep=True zeros errors internally, so this also checks that path.
    errors_dense = np.array([1e-4, 2e-4], dtype=np.float64)
    errors_coo = errors_dense.copy()

    dense_result = BoundingIntervalLinearSystem(
        dense_Ms,
        errors_dense,
        finalStep=True,
    )

    coo_result = BoundingIntervalLinearSystem(
        coo_Ms,
        errors_coo,
        finalStep=True,
    )

    assert_bounding_results_equal(coo_result, dense_result)