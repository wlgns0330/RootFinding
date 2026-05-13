import numpy as np
import itertools
from scipy import linalg as la
from math import fabs
from sparse import COO

import numpy as np
from math import fabs
from numba import njit


@njit
def _eval_quad_2d(x, y, c0, c1, c2, c3, c4, c5):
    # Chebyshev quadratic:
    # c0 + c1*T1(x) + c2*T1(y) + c3*T2(x) + c4*T1(x)*T1(y) + c5*T2(y)
    #
    # T1(x) = x
    # T2(x) = 2*x*x - 1
    k0 = c0 - c3 - c5
    k3 = 2.0 * c3
    k5 = 2.0 * c5

    return k0 + (c1 + k3 * x + c4 * y) * x + (c2 + k5 * y) * y


@njit
def _quadratic_check_2D_coo_core(coords, data, shape0, shape1, tol):
    """
    Numba-friendly core for a 2D pydata/sparse COO coefficient matrix.

    coords has shape (2, nnz)
    data has shape (nnz,)
    """

    # Extract only the quadratic coefficients.
    c0 = 0.0  # (0, 0)
    c1 = 0.0  # (1, 0)
    c2 = 0.0  # (0, 1)
    c3 = 0.0  # (2, 0)
    c4 = 0.0  # (1, 1)
    c5 = 0.0  # (0, 2)

    abs_sum = 0.0

    nnz = data.shape[0]

    for k in range(nnz):
        i = coords[0, k]
        j = coords[1, k]
        v = data[k]

        abs_sum += fabs(v)

        if i == 0 and j == 0:
            c0 += v
        elif i == 1 and j == 0:
            c1 += v
        elif i == 0 and j == 1:
            c2 += v
        elif i == 2 and j == 0:
            c3 += v
        elif i == 1 and j == 1:
            c4 += v
        elif i == 0 and j == 2:
            c5 += v

    # Same idea as dense:
    # other_sum = sum(abs(all coeffs)) - abs(quadratic coeffs) + tol
    #
    # If COO has duplicate coordinates, c0...c5 are accumulated correctly.
    # If duplicates cancel, this can overestimate other_sum, which is safe but less aggressive.
    other_sum = (
        abs_sum
        - fabs(c0)
        - fabs(c1)
        - fabs(c2)
        - fabs(c3)
        - fabs(c4)
        - fabs(c5)
        + tol
    )

    # Interior critical point.
    det = 16.0 * c3 * c5 - c4 * c4

    if det != 0.0:
        int_x = (c2 * c4 - 4.0 * c1 * c5) / det
        int_y = (c1 * c4 - 4.0 * c2 * c3) / det
    else:
        int_x = np.inf
        int_y = np.inf

    min_satisfied = False
    max_satisfied = False

    # Corners of [-1, 1]^2.
    val = _eval_quad_2d(-1.0, -1.0, c0, c1, c2, c3, c4, c5)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_2d(1.0, -1.0, c0, c1, c2, c3, c4, c5)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_2d(-1.0, 1.0, c0, c1, c2, c3, c4, c5)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_2d(1.0, 1.0, c0, c1, c2, c3, c4, c5)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    # x = -1 and x = 1 boundaries.
    # dy = 0:
    # c4*x + 4*c5*y = -c2
    if c5 != 0.0:
        cc5 = 4.0 * c5

        x = -1.0
        y = -(c2 + c4 * x) / cc5
        if -1.0 < y < 1.0:
            val = _eval_quad_2d(x, y, c0, c1, c2, c3, c4, c5)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        x = 1.0
        y = -(c2 + c4 * x) / cc5
        if -1.0 < y < 1.0:
            val = _eval_quad_2d(x, y, c0, c1, c2, c3, c4, c5)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # y = -1 and y = 1 boundaries.
    # dx = 0:
    # 4*c3*x + c4*y = -c1
    if c3 != 0.0:
        cc3 = 4.0 * c3

        y = -1.0
        x = -(c1 + c4 * y) / cc3
        if -1.0 < x < 1.0:
            val = _eval_quad_2d(x, y, c0, c1, c2, c3, c4, c5)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        y = 1.0
        x = -(c1 + c4 * y) / cc3
        if -1.0 < x < 1.0:
            val = _eval_quad_2d(x, y, c0, c1, c2, c3, c4, c5)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Interior critical point.
    if -1.0 < int_x < 1.0 and -1.0 < int_y < 1.0:
        val = _eval_quad_2d(int_x, int_y, c0, c1, c2, c3, c4, c5)
        min_satisfied = min_satisfied or val < other_sum
        max_satisfied = max_satisfied or val > -other_sum
        if min_satisfied and max_satisfied:
            return False

    # No root possible.
    return True


def quadratic_check_2D_coo(test_coeff, tol):
    """
    COO version of quadratic_check_2D.

    Expects pydata/sparse COO:
        test_coeff.coords  # shape (2, nnz)
        test_coeff.data    # shape (nnz,)
        test_coeff.shape
        test_coeff.ndim
    """

    if test_coeff.ndim != 2:
        return False

    return _quadratic_check_2D_coo_core(
        test_coeff.coords,
        test_coeff.data,
        test_coeff.shape[0],
        test_coeff.shape[1],
        tol,
    )

import numpy as np
from math import fabs
from numba import njit


@njit
def _eval_quad_3d(x, y, z,
                  c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    """
    Evaluate:

        c0
      + c1*T1(x) + c2*T1(y) + c3*T1(z)
      + c4*T1(x)*T1(y)
      + c5*T1(x)*T1(z)
      + c6*T1(y)*T1(z)
      + c7*T2(x)
      + c8*T2(y)
      + c9*T2(z)

    using T1(t) = t and T2(t) = 2*t^2 - 1.
    """

    k0 = c0 - c7 - c8 - c9
    k7 = 2.0 * c7
    k8 = 2.0 * c8
    k9 = 2.0 * c9

    return (
        k0
        + (c1 + k7 * x + c4 * y + c5 * z) * x
        + (c2 + k8 * y + c6 * z) * y
        + (c3 + k9 * z) * z
    )


@njit
def _quadratic_check_3D_coo_core(coords, data, tol):
    """
    Numba-friendly COO core.

    Parameters
    ----------
    coords : ndarray
        COO coordinates with shape (3, nnz).

    data : ndarray
        COO values with shape (nnz,).

    tol : float
        Sup-norm error tolerance.

    Returns
    -------
    bool
        True  means no root is possible.
        False means a root may be possible.
    """

    # Quadratic Chebyshev coefficients.
    #
    # c0 = coeff[0, 0, 0]
    # c1 = coeff[1, 0, 0]
    # c2 = coeff[0, 1, 0]
    # c3 = coeff[0, 0, 1]
    # c4 = coeff[1, 1, 0]
    # c5 = coeff[1, 0, 1]
    # c6 = coeff[0, 1, 1]
    # c7 = coeff[2, 0, 0]
    # c8 = coeff[0, 2, 0]
    # c9 = coeff[0, 0, 2]

    c0 = 0.0
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    c4 = 0.0
    c5 = 0.0
    c6 = 0.0
    c7 = 0.0
    c8 = 0.0
    c9 = 0.0

    abs_sum = 0.0
    nnz = data.shape[0]

    for k in range(nnz):
        i = coords[0, k]
        j = coords[1, k]
        l = coords[2, k]
        v = data[k]

        abs_sum += fabs(v)

        if i == 0 and j == 0 and l == 0:
            c0 += v
        elif i == 1 and j == 0 and l == 0:
            c1 += v
        elif i == 0 and j == 1 and l == 0:
            c2 += v
        elif i == 0 and j == 0 and l == 1:
            c3 += v
        elif i == 1 and j == 1 and l == 0:
            c4 += v
        elif i == 1 and j == 0 and l == 1:
            c5 += v
        elif i == 0 and j == 1 and l == 1:
            c6 += v
        elif i == 2 and j == 0 and l == 0:
            c7 += v
        elif i == 0 and j == 2 and l == 0:
            c8 += v
        elif i == 0 and j == 0 and l == 2:
            c9 += v

    # Same logic as dense:
    #
    # other_sum = np.sum(np.abs(test_coeff)) - sum(abs(c_i)) + tol
    #
    # Note: if the COO object contains duplicate coordinates, c0...c9 are
    # accumulated correctly, but abs_sum counts duplicate entries before
    # cancellation. That is safe but can make the check less aggressive.
    other_sum = (
        abs_sum
        - fabs(c0)
        - fabs(c1)
        - fabs(c2)
        - fabs(c3)
        - fabs(c4)
        - fabs(c5)
        - fabs(c6)
        - fabs(c7)
        - fabs(c8)
        - fabs(c9)
        + tol
    )

    # Precompute constants for derivatives.
    k7 = 2.0 * c7
    k8 = 2.0 * c8
    k9 = 2.0 * c9

    kk7 = 2.0 * k7  # 4*c7
    kk8 = 2.0 * k8  # 4*c8
    kk9 = 2.0 * k9  # 4*c9

    # Face/interior determinant pieces.
    fix_x_det = kk8 * kk9 - c6 * c6
    fix_y_det = kk7 * kk9 - c5 * c5
    fix_z_det = kk7 * kk8 - c4 * c4

    minor_1_2 = kk9 * c4 - c5 * c6
    minor_1_3 = c4 * c6 - kk8 * c5
    minor_2_3 = kk7 * c6 - c4 * c5

    det = 4.0 * c7 * fix_x_det - c4 * minor_1_2 + c5 * minor_1_3

    if det != 0.0:
        int_x = (c1 * -fix_x_det + c2 * minor_1_2 + c3 * -minor_1_3) / det
        int_y = (c1 * minor_1_2 + c2 * -fix_y_det + c3 * minor_2_3) / det
        int_z = (c1 * -minor_1_3 + c2 * minor_2_3 + c3 * -fix_z_det) / det
    else:
        int_x = np.inf
        int_y = np.inf
        int_z = np.inf

    x0 = -1.0
    x1 = 1.0
    y0 = -1.0
    y1 = 1.0
    z0 = -1.0
    z1 = 1.0

    min_satisfied = False
    max_satisfied = False

    # Helper pattern:
    # If both become true, the quadratic part overlaps the uncertainty band
    # [-other_sum, other_sum], so a root may be possible.
    #
    # Corners.
    val = _eval_quad_3d(x0, y0, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x1, y0, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x0, y1, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x0, y0, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x1, y1, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x1, y0, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x0, y1, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    val = _eval_quad_3d(x1, y1, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
    min_satisfied = min_satisfied or val < other_sum
    max_satisfied = max_satisfied or val > -other_sum
    if min_satisfied and max_satisfied:
        return False

    # Edges where x and y are fixed; solve dz = 0.
    if c9 != 0.0:
        c5x0_c3 = c5 * x0 + c3
        c6y0 = c6 * y0
        z = -(c5x0_c3 + c6y0) / kk9
        if z0 < z < z1:
            val = _eval_quad_3d(x0, y0, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c6y1 = c6 * y1
        z = -(c5x0_c3 + c6y1) / kk9
        if z0 < z < z1:
            val = _eval_quad_3d(x0, y1, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c5x1_c3 = c5 * x1 + c3
        z = -(c5x1_c3 + c6y0) / kk9
        if z0 < z < z1:
            val = _eval_quad_3d(x1, y0, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        z = -(c5x1_c3 + c6y1) / kk9
        if z0 < z < z1:
            val = _eval_quad_3d(x1, y1, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Edges where x and z are fixed; solve dy = 0.
    if c8 != 0.0:
        c6z0 = c6 * z0
        c2_c4x0 = c2 + c4 * x0
        y = -(c2_c4x0 + c6z0) / kk8
        if y0 < y < y1:
            val = _eval_quad_3d(x0, y, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c6z1 = c6 * z1
        y = -(c2_c4x0 + c6z1) / kk8
        if y0 < y < y1:
            val = _eval_quad_3d(x0, y, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c2_c4x1 = c2 + c4 * x1
        y = -(c2_c4x1 + c6z0) / kk8
        if y0 < y < y1:
            val = _eval_quad_3d(x1, y, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        y = -(c2_c4x1 + c6z1) / kk8
        if y0 < y < y1:
            val = _eval_quad_3d(x1, y, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Edges where y and z are fixed; solve dx = 0.
    if c7 != 0.0:
        c1_c4y0 = c1 + c4 * y0
        c5z0 = c5 * z0
        x = -(c1_c4y0 + c5z0) / kk7
        if x0 < x < x1:
            val = _eval_quad_3d(x, y0, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c5z1 = c5 * z1
        x = -(c1_c4y0 + c5z1) / kk7
        if x0 < x < x1:
            val = _eval_quad_3d(x, y0, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c1_c4y1 = c1 + c4 * y1
        x = -(c1_c4y1 + c5z0) / kk7
        if x0 < x < x1:
            val = _eval_quad_3d(x, y1, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        x = -(c1_c4y1 + c5z1) / kk7
        if x0 < x < x1:
            val = _eval_quad_3d(x, y1, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Faces with x fixed; solve dy = dz = 0.
    if fix_x_det != 0.0:
        c2_c4x0 = c2 + c4 * x0
        c3_c5x0 = c3 + c5 * x0
        y = (-kk9 * c2_c4x0 + c6 * c3_c5x0) / fix_x_det
        z = (c6 * c2_c4x0 - kk8 * c3_c5x0) / fix_x_det
        if y0 < y < y1 and z0 < z < z1:
            val = _eval_quad_3d(x0, y, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c2_c4x1 = c2 + c4 * x1
        c3_c5x1 = c3 + c5 * x1
        y = (-kk9 * c2_c4x1 + c6 * c3_c5x1) / fix_x_det
        z = (c6 * c2_c4x1 - kk8 * c3_c5x1) / fix_x_det
        if y0 < y < y1 and z0 < z < z1:
            val = _eval_quad_3d(x1, y, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Faces with y fixed; solve dx = dz = 0.
    if fix_y_det != 0.0:
        c1_c4y0 = c1 + c4 * y0
        c3_c6y0 = c3 + c6 * y0
        x = (-kk9 * c1_c4y0 + c5 * c3_c6y0) / fix_y_det
        z = (c5 * c1_c4y0 - kk7 * c3_c6y0) / fix_y_det
        if x0 < x < x1 and z0 < z < z1:
            val = _eval_quad_3d(x, y0, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c1_c4y1 = c1 + c4 * y1
        c3_c6y1 = c3 + c6 * y1
        x = (-kk9 * c1_c4y1 + c5 * c3_c6y1) / fix_y_det
        z = (c5 * c1_c4y1 - kk7 * c3_c6y1) / fix_y_det
        if x0 < x < x1 and z0 < z < z1:
            val = _eval_quad_3d(x, y1, z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Faces with z fixed; solve dx = dy = 0.
    if fix_z_det != 0.0:
        c1_c5z0 = c1 + c5 * z0
        c2_c6z0 = c2 + c6 * z0
        x = (-kk8 * c1_c5z0 + c4 * c2_c6z0) / fix_z_det
        y = (c4 * c1_c5z0 - kk7 * c2_c6z0) / fix_z_det
        if x0 < x < x1 and y0 < y < y1:
            val = _eval_quad_3d(x, y, z0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

        c1_c5z1 = c1 + c5 * z1
        c2_c6z1 = c2 + c6 * z1
        x = (-kk8 * c1_c5z1 + c4 * c2_c6z1) / fix_z_det
        y = (c4 * c1_c5z1 - kk7 * c2_c6z1) / fix_z_det
        if x0 < x < x1 and y0 < y < y1:
            val = _eval_quad_3d(x, y, z1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
            min_satisfied = min_satisfied or val < other_sum
            max_satisfied = max_satisfied or val > -other_sum
            if min_satisfied and max_satisfied:
                return False

    # Interior critical point.
    if x0 < int_x < x1 and y0 < int_y < y1 and z0 < int_z < z1:
        val = _eval_quad_3d(int_x, int_y, int_z,
                            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
        min_satisfied = min_satisfied or val < other_sum
        max_satisfied = max_satisfied or val > -other_sum
        if min_satisfied and max_satisfied:
            return False

    # No root possible.
    return True


def quadratic_check_3D_coo(test_coeff, tol):
    """
    COO version of quadratic_check_3D.

    Expects pydata/sparse COO:

        test_coeff.coords  # shape (3, nnz)
        test_coeff.data    # shape (nnz,)
        test_coeff.shape
        test_coeff.ndim

    Returns
    -------
    bool
        True  means no root is possible.
        False means a root may be possible.
    """

    if test_coeff.ndim != 3:
        return False

    return _quadratic_check_3D_coo_core(
        test_coeff.coords,
        test_coeff.data,
        tol,
    )

def get_fixed_vars(dim):
    """Used in quadratic_check_nd to iterate through the boundaries of the domain.

    Parameters
    ----------
    dim : int
        The dimension of the domain/system.

    Returns
    -------
    list of tuples
        A list of tuples indicating which variables to fix in each iteration,
        starting at fixing dim-1 of them and ending with fixing 1 of them. This
        intentionally excludes combinations that correspond to the corners of the
        domain and the interior extremum.
    """
    return list(itertools.chain.from_iterable(itertools.combinations(range(dim), r)\
                                             for r in range(dim-1,0,-1)))

def quadratic_check_nd_coo(test_coeff, tol):
    """
    COO version of quadratic_check_nd.

    Parameters
    ----------
    test_coeff : sparse.COO
        Sparse COO Chebyshev coefficient tensor.

    tol : float
        Bound of the sup-norm error.

    Returns
    -------
    bool
        True  if there is guaranteed to be no root in the interval.
        False otherwise.
    """

    dim = test_coeff.ndim

    # Standard Chebyshev interval [-1, 1]^dim.
    interval = [-np.ones(dim), np.ones(dim)]

    # A and B describe the derivative system:
    #
    #     A X + B = 0
    #
    # for the quadratic Chebyshev part.
    A = np.zeros((dim, dim), dtype=np.float64)
    B = np.zeros(dim, dtype=np.float64)

    const = 0.0
    pure_quad_coeff = np.zeros(dim, dtype=np.float64)

    # This is the sum of abs of all stored COO coefficients.
    # Later we subtract the absolute values of the extracted quadratic
    # coefficients to get the "other terms" bound.
    abs_sum = 0.0

    coords = test_coeff.coords
    data = test_coeff.data
    nnz = data.shape[0]

    for k in range(nnz):
        spot = coords[:, k]
        coeff = data[k]
        abs_sum += fabs(coeff)

        # The dense version pads to length >= 3 in every dimension and then
        # only inspects spots in product(range(3), repeat=dim).
        #
        # In COO, if any index is >= 3, it cannot be part of the quadratic
        # Chebyshev part.
        is_in_low_block = True
        spot_deg = 0

        for d in range(dim):
            idx = spot[d]
            if idx >= 3:
                is_in_low_block = False
                break
            spot_deg += idx

        if not is_in_low_block:
            continue

        # Only total degree 0, 1, or 2 belongs to the quadratic part.
        if spot_deg > 2:
            continue

        if spot_deg == 0:
            # Constant term: coeff[0, 0, ..., 0]
            const += coeff

        elif spot_deg == 1:
            # Linear term: one index is 1.
            for i in range(dim):
                if spot[i] != 0:
                    B[i] += coeff
                    break

        else:
            # Quadratic terms.
            where_nonzero = []
            for i in range(dim):
                if spot[i] != 0:
                    where_nonzero.append(i)

            if len(where_nonzero) == 2:
                # Cross term: x_i x_j.
                i = where_nonzero[0]
                j = where_nonzero[1]

                A[i, j] += coeff
                A[j, i] += coeff

            else:
                # Pure quadratic term: T_2(x_i).
                i = where_nonzero[0]
                pure_quad_coeff[i] += coeff

    # For T_2(x_i) = 2*x_i^2 - 1:
    #
    # quadratic contribution is c*T_2(x_i)
    # = c*(2*x_i^2 - 1)
    #
    # derivative contribution is 4*c*x_i.
    pure_quad_coeff_doubled = 2.0 * pure_quad_coeff
    A[np.diag_indices(dim)] = 2.0 * pure_quad_coeff_doubled  # 4*c_i

    # The quadratic coefficients are:
    #   const
    #   B[i]
    #   pure_quad_coeff[i]
    #   cross terms stored once in upper triangle of A
    #
    # Because A stores cross terms symmetrically, only count each cross term once.
    quad_abs_sum = fabs(const)

    for i in range(dim):
        quad_abs_sum += fabs(B[i])
        quad_abs_sum += fabs(pure_quad_coeff[i])

    for i in range(dim):
        for j in range(i + 1, dim):
            quad_abs_sum += fabs(A[i, j])

    other_sum = abs_sum - quad_abs_sum + tol

    # Horner-style evaluation of the quadratic Chebyshev part.
    k0 = const - np.sum(pure_quad_coeff)

    def eval_func(point):
        total = k0

        for i, coord in enumerate(point):
            cross_sum = 0.0
            for j in range(i + 1, dim):
                cross_sum += A[i, j] * point[j]

            total += (
                B[i]
                + pure_quad_coeff_doubled[i] * coord
                + cross_sum
            ) * coord

        return total

    fixed_vars = get_fixed_vars(dim)

    Done = False
    min_satisfied = False
    max_satisfied = False

    # Check all corners.
    for corner in itertools.product([0, 1], repeat=dim):
        point = np.array(
            [interval[j][i] for i, j in enumerate(corner)],
            dtype=np.float64,
        )

        val = eval_func(point)

        min_satisfied = min_satisfied or val < other_sum
        max_satisfied = max_satisfied or val > -other_sum

        if min_satisfied and max_satisfied:
            Done = True
            break

    # Check sides/faces/etc.
    if not Done:
        X = np.zeros(dim, dtype=np.float64)

        for fixed in fixed_vars:
            fixed = np.array(fixed, dtype=np.int64)
            unfixed = np.delete(np.arange(dim), fixed)

            A_ = A[unfixed][:, unfixed]

            # Same sign-definiteness shortcut as the dense version.
            diag = np.diag(A_)

            sign_changed = False
            for i, c in enumerate(diag[:-1]):
                if c * diag[i + 1] < 0:
                    sign_changed = True
                    break

            if sign_changed:
                continue

            # If not full rank, there is no unique critical point to check.
            if np.linalg.matrix_rank(A_, hermitian=True) != A_.shape[0]:
                continue

            fixed_A = A[unfixed][:, fixed]
            B_ = B[unfixed]

            for side in itertools.product([0, 1], repeat=len(fixed)):
                X0 = np.array(
                    [interval[j][i] for i, j in enumerate(side)],
                    dtype=np.float64,
                )

                X_ = la.solve(
                    A_,
                    -B_ - fixed_A @ X0,
                    assume_a="sym",
                )

                # Make sure the solution is inside the domain.
                in_domain = True
                for i, var in enumerate(unfixed):
                    if interval[0][var] <= X_[i] <= interval[1][var]:
                        continue
                    in_domain = False
                    break

                if not in_domain:
                    continue

                X[fixed] = X0
                X[unfixed] = X_

                val = eval_func(X)

                min_satisfied = min_satisfied or val < other_sum
                max_satisfied = max_satisfied or val > -other_sum

                if min_satisfied and max_satisfied:
                    Done = True
                    break

            if Done:
                break

        else:
            # Check interior critical point: no variables fixed.
            sign_changed = False

            for i, c in enumerate(pure_quad_coeff[:-1]):
                if c * pure_quad_coeff[i + 1] < 0:
                    sign_changed = True
                    break

            if not sign_changed:
                if np.linalg.matrix_rank(A, hermitian=True) == A.shape[0]:
                    X = la.solve(A, -B, assume_a="sym")

                    in_domain = True
                    for i in range(dim):
                        if interval[0][i] <= X[i] <= interval[1][i]:
                            continue
                        in_domain = False
                        break

                    if in_domain:
                        val = eval_func(X)

                        min_satisfied = min_satisfied or val < other_sum
                        max_satisfied = max_satisfied or val > -other_sum

                        if min_satisfied and max_satisfied:
                            Done = True

    return not Done