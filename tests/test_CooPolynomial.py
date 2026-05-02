import numpy as np
import pytest
import sparse

from yroots.polynomial import CooPolynomial, CooPower

# ---------------------------------------------------------------------
# CooPolynomial initialization tests
# ---------------------------------------------------------------------

def test_coopolynomial_init_from_dense_numpy():
    dense = np.array([
        [1.0, 0.0],
        [0.0, 3.0],
    ])

    p = CooPolynomial(dense)

    assert isinstance(p.coeff, sparse.COO)
    assert p.shape == dense.shape
    assert p.dim == 2
    assert p.coeff.nnz == 2
    assert np.allclose(p.coeff.todense(), dense)


def test_coopolynomial_init_from_sparse_coo():
    dense = np.array([
        [0.0, 2.0],
        [4.0, 0.0],
    ])
    coo = sparse.COO.from_numpy(dense)

    p = CooPolynomial(coo)

    assert isinstance(p.coeff, sparse.COO)
    assert p.shape == dense.shape
    assert p.dim == 2
    assert np.allclose(p.coeff.todense(), dense)


def test_coopolynomial_init_from_coords_data_tuple():
    coords = np.array([
        [0, 1, 2],
        [0, 2, 1],
    ])
    data = np.array([5.0, -2.0, 3.0])
    shape = (3, 3)

    p = CooPolynomial((coords, data), shape=shape)

    expected = np.zeros(shape)
    expected[0, 0] = 5.0
    expected[1, 2] = -2.0
    expected[2, 1] = 3.0

    assert isinstance(p.coeff, sparse.COO)
    assert p.shape == shape
    assert p.dim == 2
    assert np.allclose(p.coeff.todense(), expected)

    def test_coopolynomial_init_from_coords_data_infers_shape():
        coords = np.array([
            [0, 1],
            [0, 1],
        ])
        data = np.array([1.0, 2.0])

        p = CooPolynomial((coords, data))

        expected = np.array([
            [1.0, 0.0],
            [0.0, 2.0],
        ])

        assert p.shape == (2, 2)
        assert p.dim == 2
        assert np.allclose(p.coeff.todense(), expected)


def test_coopolynomial_init_from_empty_coords_data_requires_shape():
    coords = np.empty((2, 0), dtype=int)
    data = np.array([], dtype=float)

    with pytest.raises(ValueError):
        CooPolynomial((coords, data))

def test_coopolynomial_init_from_empty_coords_data_with_shape():
    coords = np.empty((2, 0), dtype=int)
    data = np.array([], dtype=float)

    p = CooPolynomial((coords, data), shape=(3, 4))

    assert p.shape == (3, 4)
    assert p.dim == 2
    assert p.nnz == 0
    assert np.allclose(p.coeff.todense(), np.zeros((3, 4)))


# ---------------------------------------------------------------------
# CooPolynomial __call__ shape-normalization tests
# ---------------------------------------------------------------------

def test_coopolynomial_call_scalar_for_1d_polynomial():
    p = CooPolynomial(np.array([1.0, 2.0, 3.0]))

    points = p(2.0)

    assert points.shape == (1, 1)
    assert np.allclose(points, np.array([[2.0]]))


def test_coopolynomial_call_1d_points_for_1d_polynomial():
    p = CooPolynomial(np.array([1.0, 2.0, 3.0]))

    points = p(np.array([1.0, 2.0, 3.0]))

    assert points.shape == (3, 1)
    assert np.allclose(points, np.array([[1.0], [2.0], [3.0]]))


def test_coopolynomial_call_single_point_for_multidim_polynomial():
    p = CooPolynomial(np.zeros((3, 4)))

    points = p(np.array([1.0, 2.0]))

    assert points.shape == (1, 2)
    assert np.allclose(points, np.array([[1.0, 2.0]]))


def test_coopolynomial_call_multiple_points_for_multidim_polynomial():
    p = CooPolynomial(np.zeros((3, 4)))

    raw_points = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    points = p(raw_points)

    assert points.shape == (2, 2)
    assert np.allclose(points, raw_points)


def test_coopolynomial_call_dimension_mismatch_raises():
    p = CooPolynomial(np.zeros((3, 4, 5)))

    with pytest.raises(ValueError):
        p(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------
# CooPolynomial equality tests
# ---------------------------------------------------------------------

def test_coopolynomial_eq_same_dense_data():
    p1 = CooPolynomial(np.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ]))

    p2 = CooPolynomial(np.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ]))

    assert p1 == p2
    assert not (p1 != p2)


def test_coopolynomial_eq_different_shape_false():
    p1 = CooPolynomial(np.zeros((2, 2)))
    p2 = CooPolynomial(np.zeros((2, 2, 1)))

    assert p1 != p2


def test_coopolynomial_eq_different_values_false():
    p1 = CooPolynomial(np.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ]))

    p2 = CooPolynomial(np.array([
        [1.0, 0.0],
        [0.0, 3.0],
    ]))

    assert p1 != p2


def test_coopolynomial_eq_coordinate_order_independent():
    coords1 = np.array([
        [0, 1],
        [0, 1],
    ])
    data1 = np.array([2.0, 5.0])

    coords2 = np.array([
        [1, 0],
        [1, 0],
    ])
    data2 = np.array([5.0, 2.0])

    p1 = CooPolynomial((coords1, data1), shape=(2, 2))
    p2 = CooPolynomial((coords2, data2), shape=(2, 2))

    assert p1 == p2


def test_coopolynomial_eq_non_polynomial_false():
    p = CooPolynomial(np.array([1.0, 2.0]))

    assert p != np.array([1.0, 2.0])