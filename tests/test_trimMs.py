import numpy as np
import pytest
from sparse import COO

# import your functions from the right module
from yroots.ChebyshevSubdivisionSolverCOO import trimMs


def test_trimMs_coo_trims_small_high_degree_slice():
    dense = np.zeros((5, 4), dtype=np.float64)

    dense[0, 0] = 1.0
    dense[2, 0] = 2.0
    dense[0, 2] = -3.0

    # Highest x-slice should be trimmed.
    dense[4, 0] = 1e-8

    # Next highest x-slice should remain.
    dense[3, 0] = 1e-2

    M = COO.from_numpy(dense)

    Ms_dense = [dense]
    Ms = [M]
    errors = np.array([1.0], dtype=np.float64)

    trimMs(Ms, errors, relApproxTol=1e-3, absApproxTol=0.0)
    trimMs(Ms_dense, errors, relApproxTol=1e-3, absApproxTol=0.0)

    assert isinstance(Ms[0], COO)
    assert Ms[0].shape == (4, 3)
    assert np.isclose(errors[0], 1.0 + 1e-8)

    np.testing.assert_allclose(Ms[0].todense(), Ms_dense[0])


def test_trimMs_coo_does_not_trim_below_degree_2():
    dense = np.zeros((4, 4), dtype=np.float64)

    dense[3, 0] = 1e-10
    dense[0, 3] = 1e-10

    M = COO.from_numpy(dense)

    Ms = [M]
    errors = np.array([1.0], dtype=np.float64)

    trimMs(Ms, errors, relApproxTol=1.0, absApproxTol=0.0)

    assert isinstance(Ms[0], COO)
    assert Ms[0].shape == (3, 3)
    assert np.isclose(errors[0], 1.0 + 2e-10)

    expected = dense[:3, :3]
    np.testing.assert_allclose(Ms[0].todense(), expected)


def test_trimMs_coo_matches_dense_result():
    dense = np.zeros((5, 5, 4), dtype=np.float64)

    # Low-degree terms.
    dense[0, 0, 0] = 1.0
    dense[1, 0, 0] = 2.0
    dense[0, 1, 0] = -1.5
    dense[0, 0, 1] = 0.7
    dense[2, 0, 0] = -0.4
    dense[0, 2, 0] = 0.3
    dense[0, 0, 2] = -0.2

    # Trimmable high-degree slices.
    dense[4, 0, 0] = 1e-8
    dense[0, 4, 0] = 2e-8
    dense[0, 0, 3] = 3e-8

    # Non-trimmable next-highest slice in dim 0.
    dense[3, 0, 0] = 1e-2

    dense_Ms = [dense.copy()]
    sparse_Ms = [COO.from_numpy(dense)]

    dense_errors = np.array([1.0], dtype=np.float64)
    sparse_errors = np.array([1.0], dtype=np.float64)

    trimMs(dense_Ms, dense_errors, relApproxTol=1e-3, absApproxTol=0.0)
    trimMs(sparse_Ms, sparse_errors, relApproxTol=1e-3, absApproxTol=0.0)

    assert sparse_Ms[0].shape == dense_Ms[0].shape
    assert np.isclose(sparse_errors[0], dense_errors[0])

    np.testing.assert_allclose(
        sparse_Ms[0].todense(),
        dense_Ms[0],
    )


def test_trimMs_coo_no_trim_when_last_slice_too_large():
    dense = np.zeros((5, 4), dtype=np.float64)

    dense[0, 0] = 1.0

    # Too large to trim with allowedErrorIncrease = 1e-3.
    dense[4, 0] = 1e-2

    M = COO.from_numpy(dense)

    Ms_dense = [dense]
    Ms = [M]
    errors = np.array([1.0], dtype=np.float64)

    trimMs(Ms, errors, relApproxTol=1e-3, absApproxTol=0.0)
    trimMs(Ms_dense, errors, relApproxTol=1e-3, absApproxTol=0.0)

    assert Ms[0].shape == (5, 3)
    assert np.isclose(errors[0], 1.0)

    np.testing.assert_allclose(Ms[0].todense(), Ms_dense[0])