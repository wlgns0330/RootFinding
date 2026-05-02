import numpy as np
from scipy.signal import convolve
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import polynomial as poly
import sparse
from math import comb
from itertools import product

def slice_top(matrix_shape):
    ''' Gets the n-d slices needed to slice a matrix into the top corner of another.

    Parameters
    ----------
    matrix_shape : tuple.
        The matrix shape of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. It is exactly the size of matrix_shape.
    '''
    slices = list()
    for i in matrix_shape:
        slices.append(slice(0,i))
    return tuple(slices)

def match_size(a,b):
    '''
    Matches the shape of two matrixes.

    Parameters
    ----------
    a, b : ndarray
        Matrixes whose size is to be matched.

    Returns
    -------
    a, b : ndarray
        Matrixes of equal size.
    '''
    new_shape = np.maximum(a.shape, b.shape)

    a_new = np.zeros(new_shape)
    a_new[slice_top(a.shape)] = a
    b_new = np.zeros(new_shape)
    b_new[slice_top(b.shape)] = b
    return a_new, b_new

############ Fast polynomial evaluation functions ############

def polyval(x, cc):
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

def polyval2(x, cc):
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

def chebval(x, cc):
    if len(cc) == 1:
        c0 = cc[0]
        c1 = np.zeros_like(c0)
    elif len(cc) == 2:
        c0 = cc[0]
        c1 = cc[1]
    else:
        x2 = 2*x
        c0 = cc[-2]
        c1 = cc[-1]
        for i in range(3, len(cc) + 1):
            tmp = c0
            c0 = cc[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

def chebval2(x, cc):
    if len(cc) == 1:
        c0 = cc[0]
        c1 = np.zeros_like(c0)
    elif len(cc) == 2:
        c0 = cc[0]
        c1 = cc[1]
    else:
        x2 = 2*x
        c0 = cc[-2]
        c1 = cc[-1]
        for i in range(3, len(cc) + 1):
            tmp = c0
            c0 = cc[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

################################################

class Polynomial(object):
    '''
    Superclass for MultiPower and MultiCheb. Contains methods and attributes
    that are applicable to both subclasses.

    Attributes
    ----------
    coeff
        The coefficient matrix represented in the object.
    dim
        The number of dimensions of the coefficient matrix
    shape
        The shape of the coefficient matrix
    lead_term
        The polynomial term with the largest total degree
    degree
        The total degree of the lead_term
    lead_coeff
        The coeff of the lead_term

    Parameters
    ----------
    coeff : ndarray
        Coefficients of the polynomial
    clean_zeros : bool
        Default is True. If True, all extra rows, columns, etc of all zeroes are
        removed from matrix of coefficients.

    Methods
    -------
    clean_coeff
        Removes extra rows, columns, etc of zeroes from end of matrix of coefficients
    match_size
        Matches the shape of two matrices.
    __call__
        Evaluates a polynomial at a certain point.
    __eq__
        Checks if two polynomials are equal.
    __ne__
        Checks if two polynomials are not equal.

    '''
    def __init__(self, coeff, clean_zeros = True):

        if isinstance(coeff,list):
            coeff = np.array(coeff)
        if isinstance(coeff,np.ndarray):
            self.coeff = coeff
            # If coeff has integer coefficients,
            # cast as numpy floats for jit compilation
            if coeff.dtype == np.int32 or coeff.dtype == np.int64:
                coeff = coeff.astype(np.float64)
        else:
            raise ValueError('Invalid input for Polynomial class object')
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.shape = self.coeff.shape
        self.jac = None

    def clean_coeff(self):
        """
        Get rid of any zeros on the outside of the coefficient matrix.
        """
        for cur_axis in range(self.coeff.ndim):
            change = True
            while change:
                change = False
                if self.coeff.shape[cur_axis] == 1:
                    continue
                slices = list()
                for i,degree in enumerate(self.coeff.shape):
                    if cur_axis == i:
                        s = slice(degree-1,degree)
                    else:
                        s = slice(0,degree)
                    slices.append(s)
                if np.sum(abs(self.coeff[tuple(slices)])) == 0:
                    self.coeff = np.delete(self.coeff,-1,axis=cur_axis)
                    change = True

    def __call__(self, points):
        '''
        Evaluates the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
         : numpy array
            valued of the polynomial at the given points
        '''
        points = np.array(points)
        if points.ndim == 0:
            points = np.array([points])

        if points.ndim == 1:
            if self.dim > 1:
                points = points.reshape(1,points.shape[0])
            else:
                points = points.reshape(points.shape[0],1)

        if points.shape[1] != self.dim:
            raise ValueError('Dimension of points does not match dimension of polynomial!')

        return points
    
    def grad(self, point):
        '''
        Evaluates the gradient of the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        grad : ndarray
            Gradient of the polynomial at the given point.
        '''
        if len(point) != self.dim:
            raise ValueError('Cannot evaluate polynomial in {} variables at point {}'\
            .format(self.dim, point))

    def __eq__(self,other):
        '''
        check if coeff matrix is the same.
        '''
        if self.shape != other.shape:
            return False
        return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        check if coeff matrix is not the same.
        '''
        return not (self == other)

    def __repr__(self):
        return str(self.coeff)
    
    def __str__(self):
        return str(self.coeff)

###############################################################################

#### MULTI_CHEB ###############################################################
class MultiCheb(Polynomial):
    """Coefficient tensor representation of a Chebyshev basis polynomial.

    Using this class instead of a Python callable function to represent a Chebyshev polynomial
    can lead to faster function evaluations during approximation.

    Examples
    --------

    To represent 4*T_2(x) + 1T_3(x) (using Chebyshev polynomials of the first kind):

    >>> f = yroots.MultiCheb([0,0,4,1])
    >>> print(f)
    [-4.   0.   5.5  0.   0.   0.   3. ]


    Parameters
    ----------
    coeff : list or numpy array
        An array containing the coefficients of the polynomial. If the polynomial is n-dimensional,
        the (i,j,...,n) index represents the term having T_i(x)*T_j(y)*....
    clean_zeros : bool
        Whether or not to remove all extra rows or columns containing only zeros. Defaults to True.

    """
    def __init__(self, coeff, clean_zeros = True):
        super(MultiCheb, self).__init__(coeff, clean_zeros)

    def __add__(self,other):
        '''Addition of two MultiCheb polynomials.

        Parameters
        ----------
        other : MultiCheb

        Returns
        -------
        MultiCheb
            The sum of the coeff of self and coeff of other.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff

        return MultiCheb(new_self + new_other)

    def __sub__(self,other):
        '''
        Subtraction of two MultiCheb polynomials.

        Parameters
        ----------
        other : MultiCheb

        Returns
        -------
        MultiCheb
            The coeff values are the result of self.coeff - other.coeff.
        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiCheb((new_self - (new_other)), clean_zeros = False)
    
    def __call__(self, points):
        '''
        Evaluates the polynomial at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        c : numpy array
            values of the polynomial at the given points
        '''
        points = super(MultiCheb, self).__call__(points)

        c = self.coeff
        n = c.ndim
        cc = c.reshape(c.shape + (1,)*points.ndim)
        c = chebval2(points[:,0],cc)
        for i in range(1,n):
            c = chebval(points[:,i],c)
        if len(c) == 1:
            return c[0]
        else:
            return c
        
    def evaluate_grid(self, xyz):
        '''
        Evaluates the Chebyshev polynomial on a grid of points, very efficiently.

        Parameters
        ----------
        xyz : array-like
            Each column contains the values for an axis. The direct product of these columns
            produces the points of the desired grid.

        Returns
        -------
        values: complex
            The polynomial evaluated at all of the points in the grid determined by
            the axis values
        '''

        xyz = super(MultiCheb, self).__call__(xyz)

        c = self.coeff
        for i in range(xyz.shape[1]):
            cc = c.reshape(c.shape + (1,)*xyz[:,i].ndim)
            c = chebval2(xyz[:,i] ,cc)

        if np.product(c.shape)==1:
            return c[0]
        else:
            return c

    def grad(self, point):
        '''
        Evaluates the gradient of the polynomial at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        out : ndarray
            Gradient of the polynomial at the given point.
        '''
        super(MultiCheb, self).__call__(point)

        out = np.empty(self.dim,dtype="complex_")
        if self.jac is None:
            jac = list()
            for i in range(self.dim):
                jac.append(cheb.chebder(self.coeff,axis=i))
            self.jac = jac
        spot = 0
        for i in self.jac:
            out[spot] = chebvalnd(point,i)
            spot+=1

        return out

###############################################################################

#### MULTI_POWER ##############################################################
class MultiPower(Polynomial):
    """Coefficient tensor representation of a power basis polynomial.

    Using this class instead of a Python callable function to represent a power basis polynomial
    can lead to faster function evaluations during approximation.

    Examples
    --------

    To represent 3x^6 + 5.5x^2 -4:

    >>> f = yroots.MultiPower([-4,0,5.5,0,0,0,3])
    >>> print(f)
    [-4.   0.   5.5  0.   0.   0.   3. ]

    To represent 0.62x^3*y - 0.11x*y + 1.03y^2 - 0.58:

    >>> f = yroots.MultiPower(np.array([[-0.58,0,1.03],[0,-0.11,0],[0,0,0],[0,0.62,0]]))
    >>> print(f)
    [[-0.58  0.    1.03]
     [ 0.   -0.11  0.  ]
     [ 0.    0.    0.  ]
     [ 0.    0.62  0.  ]]


    Parameters
    ----------
    coeff : list or numpy array
        An array containing the coefficients of the polynomial. If the polynomial is n-dimensional,
        the (i,j,...,n) index represents the term of degree i in dimension 0, degree j in dimension 1,
        and so forth.
    clean_zeros : bool
        Whether or not to remove all extra rows or columns containing only zeros. Defaults to True.

    """
    def __init__(self, coeff, clean_zeros = True):
        super(MultiPower, self).__init__(coeff, clean_zeros)

    def __add__(self,other):
        '''Addition of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower

        Returns
        -------
        MultiPower object
            The sum of the coeff of self and coeff of other.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiPower((new_self + new_other), clean_zeros = False)

    def __sub__(self,other):
        '''
        Subtraction of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower

        Returns
        -------
        MultiPower
            The coeff values are the result of self.coeff - other.coeff.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiPower((new_self - (new_other)), clean_zeros = False)

    def __mul__(self,other):
        '''
        Multiplication of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower object

        Returns
        -------
        MultiPower object
            The result of self*other.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff

        return MultiPower(convolve(new_self, new_other))
    
    def __call__(self, points):
        '''
        Evaluates the polynomial at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        __call__: complex
            value of the polynomial at the given point
        '''
        points = super(MultiPower, self).__call__(points)

        c = self.coeff
        n = c.ndim
        cc = c.reshape(c.shape + (1,)*points.ndim)
        c = polyval2(points[:,0],cc)
        for i in range(1,n):
            c = polyval(points[:,i],c)
        if len(c) == 1:
            return c[0]
        else:
            return c
    
    def evaluate_grid(self, xyz):
        '''
        Evaluates the Power polynomial on a grid of points, very efficiently.

        Parameters
        ----------
        xyz : array-like
            Each column contains the values for an axis. The direct product of these columns
            produces the points of the desired grid.

        Returns
        -------
        values: complex
            The polynomial evaluated at all of the points in the grid determined by
            the axis values
        '''

        xyz = super(MultiPower, self).__call__(xyz)

        c = self.coeff
        for i in range(xyz.shape[1]):
            cc = c.reshape(c.shape + (1,)*xyz[:,i].ndim)
            c = polyval2(xyz[:,i] ,cc)

        if np.product(c.shape)==1:
            return c[0]
        else:
            return c

    def grad(self, point):
        '''
        Evaluates the gradient of the polynomial at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        out : ndarray
            Gradient of the polynomial at the given point.
        '''
        super(MultiPower, self).__call__(point)

        out = np.empty(self.dim,dtype="complex_")
        if self.jac is None:
            jac = list()
            for i in range(self.dim):
                jac.append(poly.polyder(self.coeff,axis=i))
            self.jac = jac
        spot = 0
        for i in self.jac:
            out[spot] = polyvalnd(point,i)
            spot+=1

        return out
    def to_cheb(self):
        """ Returns the chebyshev coefficient matrix """
        def get_new_As(As):
            """ Finds the next transformation coefficients from the previous ones.
                So if x^n = sum(As[i]*T_i(x)), x^(n+1) = sum(Bs[i]*T_i(x)).
            """
            n = len(As)
            if n == 0:
                return np.array([1.])
            Bs = np.zeros(n+1)
            # Edge case if As has length 1
            if n == 1:
                Bs[1] = As[0]
                return Bs
            # Put in the first and last coeffs
            if n%2 == 0:
                Bs[0] = As[1]/2
            Bs[-1] = As[-1]/2
            # Put in the second coeff
            if n == 2:
                Bs[1] = As[0]
                return Bs
            if n%2 == 1:
                Bs[1] = As[0] + As[2]/2
            # Do all the middle coefficients, only editing the ones that shouldn't be 0.
            if n > 3:
                Bs[2+n%2:-2:2] = (As[1+n%2:-2:2] + As[3+n%2::2])/2 
            return Bs
        def to_cheb1D(coeffs):
            """Transforms to chebyshev coeficcients along the first dimension of coeffs matrix"""
            cheb_coeffs = np.zeros_like(coeffs, dtype=np.float64)
            As = []
            # Update As, then take each slice of the coefficient matrix and matrix multiply 
            # by As, and add to the cheb_coeffs matrix.
            for i,coeff in enumerate(coeffs):
                As = get_new_As(As)
                # Invoke einsum to do the right matrix multiplication in n dimensions
                cheb_coeffs[:i+1] += np.einsum("i,...->i...",As,coeff) 
                #np.expand_dims(As,axis=1)@np.array([coeff])
            return cheb_coeffs
        def to_chebND(coeffs,dim):
            """Transforms to chebyshev coefficients along the dim axis of the coeffs matrix"""
            # Get the transopse order to make the desired dim first
            order = np.array([dim] + [i for i in range(dim)] + [i for i in range(dim+1, coeffs.ndim)])
            # Then transpose with the inverted order after the transformation occurs.
            backOrder = np.zeros(coeffs.ndim, dtype = int)
            backOrder[order] = np.arange(coeffs.ndim)
            # Transpose coeffs, transform them along the first dimension, then transpose them back.
            return to_cheb1D(coeffs.transpose(order),).transpose(backOrder)
        cheb_coeffs = self.coeff
        for dim in range(self.coeff.ndim):
            # Go through each dimension and transform
            cheb_coeffs = to_chebND(cheb_coeffs,dim)
        return cheb_coeffs

        
###############################################################################

#### CHEBVALND, POLYVALND #############################################################

def chebvalnd(x,c):
    """
    Evaluate a MultiCheb object at a point x

    Parameters
    ----------
    x : ndarray
        Point to evaluate at
    c : ndarray
        Tensor of Chebyshev coefficients

    Returns
    -------
    c : float
        Value of the MultiCheb polynomial at x
    """
    x = np.array(x)
    n = c.ndim
    c = cheb.chebval(x[0],c)
    for i in range(1,n):
        c = cheb.chebval(x[i],c,tensor=False)
    return c

def polyvalnd(x,c):
    """
    Evaluate a MultiPower object at a point x

    Parameters
    ----------
    x : ndarray
        Point to evaluate at
    c : ndarray
        Tensor of Polynomial coefficients

    Returns
    -------
    c : float
        Value of the MultiPower polynomial at x
    """
    x = np.array(x)
    n = c.ndim
    c = poly.polyval(x[0],c)
    for i in range(1,n):
        c = poly.polyval(x[i],c,tensor=False)
    return c

###############################################################################

#### COO_Polynomial ###########################################################
class CooPolynomial:
    basis = None

    def __init__(self, coeffs, shape=None, clean_coeff=True):
        """
        coeffs:
            Can be one of:
            - sparse.COO
            - dense numpy array
            - (coords, data) tuple
        shape:
            Required when coeffs is a (coords, data) tuple unless shape can be inferred.
        prune_zeros:
            If True, remove explicitly stored zeros.
        """
        if isinstance(coeffs, sparse.COO):
            coo = coeffs
        elif isinstance(coeffs, tuple) and len(coeffs) == 2:
            coords, data = coeffs
            coords = np.asarray(coords)
            data = np.asarray(data)

            # Check for any errors in COO data shapes
            if coords.ndim != 2: raise ValueError("coords must be a 2D array with shape (ndim, nnz).")
            if data.ndim != 1: raise ValueError("data must be a 1D array with shape (nnz,).")
            if coords.shape[1] != data.shape[0]: raise ValueError("coords.shape[1] must equal data.shape[0].")
            if shape is None:
                if coords.shape[1] == 0: raise ValueError("shape is required when coords/data are empty.")
                shape = tuple(coords.max(axis=1) + 1)

            coo = sparse.COO(coords, data, shape=shape)
        else:
            coo = sparse.COO.from_numpy(np.asarray(coeffs))

        if clean_coeff and 0 in coo.data:
            coo = coo.copy()
            coo.data = np.asarray(coo.data)
            mask = coo.data != 0
            coo.coords = coo.coords[:, mask]
            coo.data = coo.data[mask]

        self.coeff = coo
        self.shape = coo.shape
        self.dim = coo.ndim
        self.dtype = coo.dtype

    def __call__(self, points):
        """
        Evaluates the polynomial at the given point.

        This method is overridden by the CooPower and CooCheb classes,
        so this definition only checks if the polynomial can be evaluated
        at the given point.

        Parameters
        ----------
        points : array-like
            The points at which to evaluate the polynomial.

        Returns
        -------
        points : numpy array
            Normalized array of points with shape (num_points, dim).
        """
        points = np.array(points)

        if points.ndim == 0:
            points = np.array([points])

        if points.ndim == 1:
            if self.dim > 1:
                points = points.reshape(1, points.shape[0])
            else:
                points = points.reshape(points.shape[0], 1)

        if points.shape[1] != self.dim:
            raise ValueError(
                "Dimension of points does not match dimension of polynomial!"
            )

        return points

    def __eq__(self, other):
        if not isinstance(other, CooPolynomial):
            return False
        
        if self.shape != other.shape:
            return False
        
        diff = self.coeff - other.coeff
        return diff.nnz == 0 or np.allclose(diff.data, 0)

    def __ne__(self, other):
        """
        Check if coeff matrix is not the same.
        """
        return not (self == other)

    def __repr__(self):
        return str(self.coeff)

    def __str__(self):
        return str(self.coeff)

    @property
    def nnz(self):
        return self.coeff.nnz

    @property
    def density(self):
        return self.nnz / np.prod(self.shape)

    def to_dense(self):
        return self.coeff.todense()

    def copy(self):
        return type(self)(self.coeff.copy())
    
###############################################################################

#### COO_Power ###########################################################
class CooPower(CooPolynomial):
    basis = "power"

    def __call__(self, points):
        points = super().__call__(points)

        coords = self.coeff.coords
        data = self.coeff.data

        num_points = points.shape[0]
        values = np.zeros(num_points, dtype=np.result_type(points, data))

        if self.nnz == 0:
            return values[0] if num_points == 1 else values

        powers = []
        for dim in range(self.dim):
            max_power = self.shape[dim] - 1
            dim_powers = np.empty(
                (num_points, max_power + 1),
                dtype=np.result_type(points, data),
            )
            dim_powers[:, 0] = 1

            for k in range(1, max_power + 1):
                dim_powers[:, k] = dim_powers[:, k - 1] * points[:, dim]

            powers.append(dim_powers)

        for j in range(self.nnz):
            term = data[j]

            for dim in range(self.dim):
                term = term * powers[dim][:, coords[dim, j]]

            values += term

        return values[0] if len(values) == 1 else values

    def evaluate_grid(self, xyz):
        """
        Evaluate sparse power polynomial on a tensor-product grid.

        xyz should have shape (num_axis_values, dim), matching the existing
        MultiPower convention where each column contains values for one axis.
        """
        xyz = super().__call__(xyz)

        coords = self.coeff.coords
        data = self.coeff.data

        axis_values = [xyz[:, dim] for dim in range(self.dim)]
        grid_shape = tuple(len(axis) for axis in axis_values)

        values = np.zeros(grid_shape, dtype=np.result_type(xyz, data))

        if self.nnz == 0:
            return values

        powers = []
        for dim, axis in enumerate(axis_values):
            max_power = self.shape[dim] - 1
            dim_powers = np.empty(
                (len(axis), max_power + 1),
                dtype=np.result_type(xyz, data),
            )

            dim_powers[:, 0] = 1

            for k in range(1, max_power + 1):
                dim_powers[:, k] = dim_powers[:, k - 1] * axis

            powers.append(dim_powers)

        for j in range(self.nnz):
            term = data[j]

            for dim in range(self.dim):
                axis_term = powers[dim][:, coords[dim, j]]

                shape = [1] * self.dim
                shape[dim] = len(axis_term)

                term = term * axis_term.reshape(shape)

            values += term

        if np.prod(values.shape) == 1:
            return values.reshape(-1)[0]

        return values

    @staticmethod
    def _coalesce_coords_data(coords, data, shape, result_dtype, tol=0.0):
        """
        Merge duplicate COO coordinates by summing their coefficients.
        coords shape: (dim, nnz)
        data shape: (nnz,)
        """
        if data.size == 0:
            return coords, data.astype(result_dtype, copy=False)

        linear = np.ravel_multi_index(coords, shape)

        order = np.argsort(linear)
        linear = linear[order]
        data = data[order]

        unique_linear, starts = np.unique(linear, return_index=True)
        summed = np.add.reduceat(data, starts).astype(result_dtype, copy=False)

        if tol == 0.0:
            keep = summed != 0
        else:
            keep = np.abs(summed) > tol

        unique_linear = unique_linear[keep]
        summed = summed[keep]

        if summed.size == 0:
            empty_coords = np.empty((coords.shape[0], 0), dtype=np.int64)
            empty_data = np.array([], dtype=result_dtype)
            return empty_coords, empty_data

        new_coords = np.array(np.unravel_index(unique_linear, shape), dtype=np.int64)
        return new_coords, summed

    @staticmethod
    def _power_to_cheb_1d(n, dtype=float):
        """
        Convert x^n into Chebyshev coefficients.

        Returns
        -------
        terms : list[tuple[int, scalar]]
            List of (cheb_degree, coefficient) pairs.
        """
        if n < 0:
            raise ValueError("Power must be nonnegative.")

        if n == 0:
            return [(0, dtype(1))]

        terms = []

        # k has same parity as n: n, n-2, n-4, ...
        for k in range(n, -1, -2):
            if k == 0:
                coeff = comb(n, n // 2) / (2 ** n)
            else:
                coeff = comb(n, (n - k) // 2) / (2 ** (n - 1))

            terms.append((k, dtype(coeff)))

        return terms

    def to_cheb(self, preserve_shape=False):
        """
        Convert a sparse power-basis polynomial to sparse Chebyshev basis.
        """
        coords = np.asarray(self.coeff.coords, dtype=np.int64)
        data = np.asarray(self.coeff.data)

        result_dtype = np.result_type(data, float)
        scalar_dtype = np.dtype(result_dtype).type

        if self.nnz == 0:
            empty_coords = np.empty((self.dim, 0), dtype=np.int64)
            empty_data = np.array([], dtype=result_dtype)
            return CooCheb((empty_coords, empty_data), shape=self.shape)

        power_cache = {}

        def get_power_terms(power):
            power = int(power)
            if power not in power_cache:
                terms = self._power_to_cheb_1d(power, dtype=scalar_dtype)

                # Store as NumPy arrays instead of list[tuple].
                idxs = np.array([t[0] for t in terms], dtype=np.int64)
                vals = np.array([t[1] for t in terms], dtype=result_dtype)

                power_cache[power] = (idxs, vals)

            return power_cache[power]

        coord_blocks = []
        data_blocks = []

        for j in range(self.nnz):
            power_multi_index = coords[:, j]
            coefficient = data[j]

            per_dim = [get_power_terms(power) for power in power_multi_index]

            idx_arrays = [p[0] for p in per_dim]
            val_arrays = [p[1] for p in per_dim]

            # Number of expanded Chebyshev terms for this monomial.
            term_count = 1
            for arr in idx_arrays:
                term_count *= arr.size

            if term_count == 1:
                # Fast path for constants.
                block_coords = np.zeros((self.dim, 1), dtype=np.int64)
                block_data = np.array([coefficient], dtype=result_dtype)
            else:
                # Build tensor-product coordinate grid.
                coord_grids = np.meshgrid(*idx_arrays, indexing="ij")
                block_coords = np.vstack([g.ravel() for g in coord_grids]).astype(np.int64, copy=False)

                # Build tensor-product coefficient grid.
                coeff_grid = coefficient
                for d, vals in enumerate(val_arrays):
                    shape = [1] * self.dim
                    shape[d] = vals.size
                    coeff_grid = coeff_grid * vals.reshape(shape)

                block_data = np.asarray(coeff_grid, dtype=result_dtype).ravel()

            coord_blocks.append(block_coords)
            data_blocks.append(block_data)

        raw_coords = np.hstack(coord_blocks)
        raw_data = np.concatenate(data_blocks).astype(result_dtype, copy=False)

        # Coalesce using self.shape because Cheb degrees never exceed power degrees.
        new_coords, new_data = self._coalesce_coords_data(raw_coords, raw_data, self.shape, result_dtype, tol=0.0)

        if new_data.size == 0:
            empty_coords = np.empty((self.dim, 0), dtype=np.int64)
            empty_data = np.array([], dtype=result_dtype)
            return CooCheb((empty_coords, empty_data), shape=self.shape)

        if preserve_shape:
            shape = self.shape
        else:
            shape = tuple(new_coords.max(axis=1) + 1)

        return CooCheb((new_coords, new_data), shape=shape)

###############################################################################

#### COO_Cheb ###########################################################

class CooCheb(CooPolynomial):
    basis = "cheb"

    def __call__(self, points):
        """
        Evaluate a sparse Chebyshev-basis polynomial at arbitrary points.

        Parameters
        ----------
        points : array-like
            Shape can be:
            - scalar for one-dimensional polynomial
            - (dim,) for one point
            - (num_points, dim) for many points

        Returns
        -------
        values : scalar or numpy.ndarray
            Polynomial values at the given points.
        """
        points = super().__call__(points)

        coords = self.coeff.coords      # shape: (dim, nnz)
        data = self.coeff.data          # shape: (nnz,)

        num_points = points.shape[0]
        result_dtype = np.result_type(points, data)
        values = np.zeros(num_points, dtype=result_dtype)

        if self.nnz == 0:
            return values[0] if num_points == 1 else values

        # cheb_values[dim][:, k] stores T_k(points[:, dim])
        cheb_values = []

        for dim in range(self.dim):
            max_degree = int(coords[dim].max())

            dim_values = np.empty((num_points, max_degree + 1), dtype=result_dtype)

            # T_0(x) = 1
            dim_values[:, 0] = 1

            if max_degree >= 1:
                # T_1(x) = x
                x = points[:, dim]
                dim_values[:, 1] = x

                # T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
                x2 = 2 * x
                for k in range(2, max_degree + 1):
                    dim_values[:, k] = (
                        x2 * dim_values[:, k - 1]
                        - dim_values[:, k - 2]
                    )

            cheb_values.append(dim_values)

        for j in range(self.nnz):
            term = data[j]

            for dim in range(self.dim):
                degree = coords[dim, j]
                term = term * cheb_values[dim][:, degree]

            values += term

        return values[0] if num_points == 1 else values

    def evaluate_grid(self, xyz):
        """
        Evaluate a sparse Chebyshev-basis polynomial on a tensor-product grid.

        Parameters
        ----------
        xyz : array-like
            Shape should be (num_axis_values, dim), matching the existing
            MultiCheb convention where each column contains the values for
            one axis.

        Returns
        -------
        values : scalar or numpy.ndarray
            Polynomial evaluated on the full tensor-product grid.
        """
        xyz = super().__call__(xyz)

        coords = self.coeff.coords
        data = self.coeff.data

        axis_values = [xyz[:, dim] for dim in range(self.dim)]
        grid_shape = tuple(len(axis) for axis in axis_values)

        result_dtype = np.result_type(xyz, data)
        values = np.zeros(grid_shape, dtype=result_dtype)

        if self.nnz == 0:
            return values.reshape(-1)[0] if np.prod(values.shape) == 1 else values

        # cheb_values[dim][:, k] stores T_k(axis_values[dim])
        cheb_values = []

        for dim, axis in enumerate(axis_values):
            max_degree = int(coords[dim].max())
            dim_values = np.empty((len(axis), max_degree + 1), dtype=result_dtype)

            # T_0(x) = 1
            dim_values[:, 0] = 1

            if max_degree >= 1:
                # T_1(x) = x
                dim_values[:, 1] = axis

                # T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
                axis2 = 2 * axis
                for k in range(2, max_degree + 1):
                    dim_values[:, k] = (
                        axis2 * dim_values[:, k - 1]
                        - dim_values[:, k - 2]
                    )

            cheb_values.append(dim_values)

        # Precompute broadcast shapes:
        # dim 0 -> (len(x), 1, 1, ...)
        # dim 1 -> (1, len(y), 1, ...)
        # etc.
        broadcast_shapes = []

        for dim, axis in enumerate(axis_values):
            shape = [1] * self.dim
            shape[dim] = len(axis)
            broadcast_shapes.append(tuple(shape))

        for j in range(self.nnz):
            term = data[j]

            for dim in range(self.dim):
                degree = coords[dim, j]
                axis_term = cheb_values[dim][:, degree]
                term = term * axis_term.reshape(broadcast_shapes[dim])

            values += term

        if np.prod(values.shape) == 1:
            return values.reshape(-1)[0]

        return values