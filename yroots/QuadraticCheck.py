"""Quadratic subinterval checks used by the Chebyshev subdivision solver.

Each ``quadratic_check_*`` routine extracts the quadratic part of a Chebyshev
coefficient tensor and bounds it against the absolute sum of the remaining
terms. If those bounds rule out a zero on the current subinterval, the check
returns ``True`` so the solver can discard the box. The :func:`quadratic_check`
dispatcher picks the dimension-specialized routine.
"""
import numpy as np
import itertools
from scipy import linalg as la
from math import fabs
from numba import njit

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

def quadratic_check(test_coeff, tol, nd_check=False):
    """Dispatch to the dimension-specialized quadratic check.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check.
    tol : float
        The bound of the sup norm error of the Chebyshev approximation.
    nd_check : bool
        If True, always use :func:`quadratic_check_nd` regardless of dimension. Defaults to False,
        which dispatches to :func:`quadratic_check_2D` or :func:`quadratic_check_3D` when possible.

    Returns
    -------
    bool
        True if the polynomial is guaranteed to have no zero on the unit box, False otherwise.
    """
    if test_coeff.ndim == 2 and not nd_check:
        return quadratic_check_2D(test_coeff, tol)
    elif test_coeff.ndim == 3 and not nd_check:
        return quadratic_check_3D(test_coeff, tol)
    else:
        return quadratic_check_nd(test_coeff, tol)

@njit
def quadratic_check_2D(test_coeff, tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. There can't be a root if min(extreme_values) > other_sum or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    True if the function is guaranteed to never be zero in the interval. False otherwise.
    """
    # Get the coefficients of the quadratic part; default to 0 when out of range.
    shape = test_coeff.shape
    c0 = test_coeff[0, 0]
    c1 = test_coeff[1, 0] if shape[0] > 1 else 0.0
    c2 = test_coeff[0, 1] if shape[1] > 1 else 0.0
    c3 = test_coeff[2, 0] if shape[0] > 2 else 0.0
    c4 = test_coeff[1, 1] if (shape[0] > 1 and shape[1] > 1) else 0.0
    c5 = test_coeff[0, 2] if shape[1] > 2 else 0.0

    # Sum of absolute values of everything except the six low-degree terms.
    other_sum = (np.sum(np.abs(test_coeff))
                 - abs(c0) - abs(c1) - abs(c2) - abs(c3) - abs(c4) - abs(c5)
                 + tol)

    # Horner form of c0 + c1*T_1(x) + c2*T_1(y) + c3*T_2(x) + c4*T_1(x)*T_1(y) + c5*T_2(y).
    k0 = c0 - c3 - c5
    k3 = 2 * c3
    k5 = 2 * c5

    min_satisfied = False
    max_satisfied = False

    # Four corners.
    for xs in (-1.0, 1.0):
        for ys in (-1.0, 1.0):
            e = k0 + (c1 + k3 * xs + c4 * ys) * xs + (c2 + k5 * ys) * ys
            if e < other_sum:
                min_satisfied = True
            if e > -other_sum:
                max_satisfied = True
            if min_satisfied and max_satisfied:
                return False

    # x-constant edges (dy=0 gives y = -(c2 + c4*x)/(4*c5)).
    if c5 != 0.0:
        cc5 = 4 * c5
        for xs in (-1.0, 1.0):
            ys = -(c2 + c4 * xs) / cc5
            if -1.0 < ys < 1.0:
                e = k0 + (c1 + k3 * xs + c4 * ys) * xs + (c2 + k5 * ys) * ys
                if e < other_sum:
                    min_satisfied = True
                if e > -other_sum:
                    max_satisfied = True
                if min_satisfied and max_satisfied:
                    return False

    # y-constant edges (dx=0 gives x = -(c1 + c4*y)/(4*c3)).
    if c3 != 0.0:
        cc3 = 4 * c3
        for ys in (-1.0, 1.0):
            xs = -(c1 + c4 * ys) / cc3
            if -1.0 < xs < 1.0:
                e = k0 + (c1 + k3 * xs + c4 * ys) * xs + (c2 + k5 * ys) * ys
                if e < other_sum:
                    min_satisfied = True
                if e > -other_sum:
                    max_satisfied = True
                if min_satisfied and max_satisfied:
                    return False

    # Interior extremum (only when the Hessian is non-singular).
    det = 16 * c3 * c5 - c4 * c4
    if det != 0.0:
        int_x = (c2 * c4 - 4 * c1 * c5) / det
        int_y = (c1 * c4 - 4 * c2 * c3) / det
        if -1.0 < int_x < 1.0 and -1.0 < int_y < 1.0:
            e = k0 + (c1 + k3 * int_x + c4 * int_y) * int_x + (c2 + k5 * int_y) * int_y
            if e < other_sum:
                min_satisfied = True
            if e > -other_sum:
                max_satisfied = True
            if min_satisfied and max_satisfied:
                return False

    return True

@njit(cache=True)
def quadratic_check_3D(test_coeff, tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.  There can't be a root if min(extreme_values) > other_sum or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    bool
        True if the function is guaranteed to never be zero in the unit box, False otherwise.
    """
    shape = test_coeff.shape
    c0 = test_coeff[0, 0, 0]
    c1 = test_coeff[1, 0, 0] if shape[0] > 1 else 0.0
    c2 = test_coeff[0, 1, 0] if shape[1] > 1 else 0.0
    c3 = test_coeff[0, 0, 1] if shape[2] > 1 else 0.0
    c4 = test_coeff[1, 1, 0] if (shape[0] > 1 and shape[1] > 1) else 0.0
    c5 = test_coeff[1, 0, 1] if (shape[0] > 1 and shape[2] > 1) else 0.0
    c6 = test_coeff[0, 1, 1] if (shape[1] > 1 and shape[2] > 1) else 0.0
    c7 = test_coeff[2, 0, 0] if shape[0] > 2 else 0.0
    c8 = test_coeff[0, 2, 0] if shape[1] > 2 else 0.0
    c9 = test_coeff[0, 0, 2] if shape[2] > 2 else 0.0

    other_sum = (np.sum(np.abs(test_coeff))
                 - abs(c0) - abs(c1) - abs(c2) - abs(c3) - abs(c4)
                 - abs(c5) - abs(c6) - abs(c7) - abs(c8) - abs(c9)
                 + tol)

    # Horner form of c0 + c1*x + c2*y + c3*z + c4*xy + c5*xz + c6*yz
    #                + c7*T_2(x) + c8*T_2(y) + c9*T_2(z).
    k0 = c0 - c7 - c8 - c9
    k7 = 2 * c7
    k8 = 2 * c8
    k9 = 2 * c9
    kk7 = 2 * k7   # 4*c7
    kk8 = 2 * k8   # 4*c8
    kk9 = 2 * k9   # 4*c9

    def eval_func(x, y, z):
        return (k0
                + (c1 + k7 * x + c4 * y + c5 * z) * x
                + (c2 + k8 * y + c6 * z) * y
                + (c3 + k9 * z) * z)

    min_satisfied = False
    max_satisfied = False

    # 8 corners.
    for xs in (-1.0, 1.0):
        for ys in (-1.0, 1.0):
            for zs in (-1.0, 1.0):
                e = eval_func(xs, ys, zs)
                if e < other_sum:
                    min_satisfied = True
                if e > -other_sum:
                    max_satisfied = True
                if min_satisfied and max_satisfied:
                    return False

    # Edges: two coordinates fixed, one free.
    # (x,y) fixed, dz=0 gives z = -(c3 + c5*x + c6*y)/(4*c9).
    if c9 != 0.0:
        for xs in (-1.0, 1.0):
            for ys in (-1.0, 1.0):
                zs = -(c3 + c5 * xs + c6 * ys) / kk9
                if -1.0 < zs < 1.0:
                    e = eval_func(xs, ys, zs)
                    if e < other_sum:
                        min_satisfied = True
                    if e > -other_sum:
                        max_satisfied = True
                    if min_satisfied and max_satisfied:
                        return False

    # (x,z) fixed, dy=0 gives y = -(c2 + c4*x + c6*z)/(4*c8).
    if c8 != 0.0:
        for xs in (-1.0, 1.0):
            for zs in (-1.0, 1.0):
                ys = -(c2 + c4 * xs + c6 * zs) / kk8
                if -1.0 < ys < 1.0:
                    e = eval_func(xs, ys, zs)
                    if e < other_sum:
                        min_satisfied = True
                    if e > -other_sum:
                        max_satisfied = True
                    if min_satisfied and max_satisfied:
                        return False

    # (y,z) fixed, dx=0 gives x = -(c1 + c4*y + c5*z)/(4*c7).
    if c7 != 0.0:
        for ys in (-1.0, 1.0):
            for zs in (-1.0, 1.0):
                xs = -(c1 + c4 * ys + c5 * zs) / kk7
                if -1.0 < xs < 1.0:
                    e = eval_func(xs, ys, zs)
                    if e < other_sum:
                        min_satisfied = True
                    if e > -other_sum:
                        max_satisfied = True
                    if min_satisfied and max_satisfied:
                        return False

    # Faces: one coordinate fixed, two free (2x2 solve on each face).
    fix_x_det = kk8 * kk9 - c6 * c6
    fix_y_det = kk7 * kk9 - c5 * c5
    fix_z_det = kk7 * kk8 - c4 * c4

    if fix_x_det != 0.0:
        for xs in (-1.0, 1.0):
            rhs2 = c2 + c4 * xs
            rhs3 = c3 + c5 * xs
            ys = (-kk9 * rhs2 + c6 * rhs3) / fix_x_det
            zs = (c6 * rhs2 - kk8 * rhs3) / fix_x_det
            if -1.0 < ys < 1.0 and -1.0 < zs < 1.0:
                e = eval_func(xs, ys, zs)
                if e < other_sum:
                    min_satisfied = True
                if e > -other_sum:
                    max_satisfied = True
                if min_satisfied and max_satisfied:
                    return False

    if fix_y_det != 0.0:
        for ys in (-1.0, 1.0):
            rhs1 = c1 + c4 * ys
            rhs3 = c3 + c6 * ys
            xs = (-kk9 * rhs1 + c5 * rhs3) / fix_y_det
            zs = (c5 * rhs1 - kk7 * rhs3) / fix_y_det
            if -1.0 < xs < 1.0 and -1.0 < zs < 1.0:
                e = eval_func(xs, ys, zs)
                if e < other_sum:
                    min_satisfied = True
                if e > -other_sum:
                    max_satisfied = True
                if min_satisfied and max_satisfied:
                    return False

    if fix_z_det != 0.0:
        for zs in (-1.0, 1.0):
            rhs1 = c1 + c5 * zs
            rhs2 = c2 + c6 * zs
            xs = (-kk8 * rhs1 + c4 * rhs2) / fix_z_det
            ys = (c4 * rhs1 - kk7 * rhs2) / fix_z_det
            if -1.0 < xs < 1.0 and -1.0 < ys < 1.0:
                e = eval_func(xs, ys, zs)
                if e < other_sum:
                    min_satisfied = True
                if e > -other_sum:
                    max_satisfied = True
                if min_satisfied and max_satisfied:
                    return False

    # Interior extremum (only when the Hessian is non-singular).
    minor_1_2 = kk9 * c4 - c5 * c6
    minor_1_3 = c4 * c6 - kk8 * c5
    minor_2_3 = kk7 * c6 - c4 * c5
    det = 4 * c7 * fix_x_det - c4 * minor_1_2 + c5 * minor_1_3
    if det != 0.0:
        int_x = (c1 * -fix_x_det + c2 * minor_1_2 + c3 * -minor_1_3) / det
        int_y = (c1 * minor_1_2 + c2 * -fix_y_det + c3 * minor_2_3) / det
        int_z = (c1 * -minor_1_3 + c2 * minor_2_3 + c3 * -fix_z_det) / det
        if -1.0 < int_x < 1.0 and -1.0 < int_y < 1.0 and -1.0 < int_z < 1.0:
            e = eval_func(int_x, int_y, int_z)
            if e < other_sum:
                min_satisfied = True
            if e > -other_sum:
                max_satisfied = True
            if min_satisfied and max_satisfied:
                return False


def quadratic_check_nd(test_coeff, tol):
    """N-dimensional specialization of :func:`quadratic_check`.

    Finds the min of the absolute value of the quadratic part and compares it to the sum of
    the remaining terms. There can't be a root if ``min(extreme_values) > other_sum`` or if
    ``max(extreme_values) < -other_sum``. Short-circuits as soon as one value below
    ``other_sum`` and one value above ``-other_sum`` are found.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check.
    tol : float
        The bound of the sup norm error of the Chebyshev approximation.

    Returns
    -------
    bool
        True if the polynomial is guaranteed to have no zero on the unit box, False otherwise.
    """
    #get the dimension and make sure the coeff tensor has all the right
    # quadratic coeff spots, set to zero if necessary
    dim = test_coeff.ndim
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')
    interval = [-np.ones(dim), np.ones(dim)]

    #Possible extrema of quadratic part are where D_xk = 0 for some subset of the variables xk
    # with the other variables are fixed to a boundary value
    #Dxk = c[0,...,0,1,0,...0] (k-spot is 1) + 4c[0,...,0,2,0,...0] xk (k-spot is 2)
    #       + \Sum_{j\neq k} xj c[0,...,0,1,0,...,0,1,0,...0] (k and j spot are 1)
    #This gives a symmetric system of equations AX+B = 0
    #We will fix different columns of X each time, resulting in slightly different
    #systems, but storing A and B now will be helpful later

    #pull out coefficients we care about
    quad_coeff = np.zeros([3]*dim)
    #A and B are arrays for slicing
    A = np.zeros([dim,dim])
    B = np.zeros(dim)
    pure_quad_coeff = [0]*dim
    for spot in itertools.product(range(3),repeat=dim):
        spot_deg = sum(spot)
        if spot_deg == 1:
            #coeff of linear terms
            i = [idx for idx in range(dim) if spot[idx]!= 0][0]
            B[i] = test_coeff[spot].copy()
            quad_coeff[spot] = test_coeff[spot]
            test_coeff[spot] = 0
        elif spot_deg == 0:
            #constant term
            const = test_coeff[spot].copy()
            quad_coeff[spot] = const
            test_coeff[spot] = 0
        elif spot_deg < 3:
            where_nonzero = [idx for idx in range(dim) if spot[idx]!= 0]
            if len(where_nonzero) == 2:
                #coeff of cross terms
                i,j = where_nonzero
                #with symmetric matrices, we only need to store the lower part
                A[j,i] = test_coeff[spot].copy()
                A[i,j] = A[j,i]
                #todo: see if we can store this in only one half of A

            else:
                #coeff of pure quadratic terms
                i = where_nonzero[0]
                pure_quad_coeff[i] = test_coeff[spot].copy()
            quad_coeff[spot] = test_coeff[spot]
            test_coeff[spot] = 0
    pure_quad_coeff_doubled = [p*2 for p in pure_quad_coeff]
    A[np.diag_indices(dim)] = [p*2 for p in pure_quad_coeff_doubled]

    #create a poly object for evals
    k0 = const - sum(pure_quad_coeff)
    def eval_func(point):
        "fast evaluation of quadratic chebyshev polynomials using horner's algorithm"
        _sum = k0
        for i,coord in enumerate(point):
            _sum += (B[i] + pure_quad_coeff_doubled[i]*coord + \
                     sum([A[i,j]*point[j] for j in range(i+1,dim)])) * coord
        return _sum

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff)) + tol

    #iterator for sides
    fixed_vars = get_fixed_vars(dim)

    Done = False
    min_satisfied, max_satisfied = False,False
    #fix all variables--> corners
    for corner in itertools.product([0,1],repeat=dim):
        #j picks if upper/lower bound. i is which var
        eval = eval_func([interval[j][i] for i,j in enumerate(corner)])
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            Done = True
            break
    #need to check sides/interior
    if not Done:
        X = np.zeros(dim)
        for fixed in fixed_vars:
            #fixed some variables --> "sides"
            #we only care about the equations from the unfixed variables
            fixed = np.array(fixed)
            unfixed = np.delete(np.arange(dim), fixed)
            A_ = A[unfixed][:,unfixed]
            #if diagonal entries change sign, can't be definite
            diag = np.diag(A_)
            for i,c in enumerate(diag[:-1]):
                #sign change?
                if c*diag[i+1]<0:
                    break
            #if no sign change, can find extrema
            else:
                #not full rank --> no soln
                if np.linalg.matrix_rank(A_,hermitian=True) == A_.shape[0]:
                    fixed_A = A[unfixed][:,fixed]
                    B_ = B[unfixed]
                    for side in itertools.product([0,1],repeat=len(fixed)):
                        X0 = np.array([interval[j][i] for i,j in enumerate(side)])
                        X_ = la.solve(A_, -B_-fixed_A@X0, assume_a='sym')
                        #make sure it's in the domain
                        for i,var in enumerate(unfixed):
                            if interval[0][var] <= X_[i] <= interval[1][var]:
                                continue
                            else:
                                break
                        else:
                            X[fixed] = X0
                            X[unfixed] = X_
                            eval = eval_func(X)
                            min_satisfied = min_satisfied or eval < other_sum
                            max_satisfied = max_satisfied or eval > -other_sum
                            if min_satisfied and max_satisfied:
                                Done = True
                                break
            if Done:
                break
        else:
            #fix no vars--> interior
            #if diagonal entries change sign, can't be definite
            for i,c in enumerate(pure_quad_coeff[:-1]):
                #sign change?
                if c*pure_quad_coeff[i+1]<0:
                    break
            #if no sign change, can find extrema
            else:
                #not full rank --> no soln
                if np.linalg.matrix_rank(A,hermitian=True) == A.shape[0]:
                    X = la.solve(A, -B, assume_a='sym')
                    #make sure it's in the domain
                    for i in range(dim):
                        if interval[0][i] <= X[i] <= interval[1][i]:
                            continue
                        else:
                            break
                    else:
                        eval = eval_func(X)
                        min_satisfied = min_satisfied or eval < other_sum
                        max_satisfied = max_satisfied or eval > -other_sum
                        if min_satisfied and max_satisfied:
                            Done = True
        #no root
    return not Done