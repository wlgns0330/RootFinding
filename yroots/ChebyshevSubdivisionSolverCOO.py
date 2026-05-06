import numpy as np
from sparse import COO, matmul
from numba import njit

@njit
def build_cheb_transform_matrix(n, alpha, beta):
    C = np.zeros((n, n))

    # Column 0: T_0(alpha*x + beta) = T_0(x)
    C[0, 0] = 1.0

    if n == 1:
        return C

    # Column 1: T_1(alpha*x + beta) = beta*T_0(x) + alpha*T_1(x)
    C[0, 1] = beta
    C[1, 1] = alpha

    arr1 = C[:, 0].copy()
    arr2 = C[:, 1].copy()
    arr3 = np.zeros(n)

    maxRow = 2

    for col in range(2, n):
        arr3[:] = 0.0

        arr3[0] = -arr1[0] + alpha * arr2[1] + 2 * beta * arr2[0]

        if maxRow > 2:
            arr3[1] = (
                -arr1[1]
                + alpha * (2 * arr2[0] + arr2[2])
                + 2 * beta * arr2[1]
            )

        for i in range(2, maxRow - 1):
            arr3[i] = (
                -arr1[i]
                + alpha * (arr2[i - 1] + arr2[i + 1])
                + 2 * beta * arr2[i]
            )

        i = maxRow - 1
        arr3[i] = (
            -arr1[i]
            + (2 if i == 1 else 1) * alpha * arr2[i - 1]
            + 2 * beta * arr2[i]
        )

        finalVal = alpha * arr2[i]

        if abs(finalVal) > 1e-16 and maxRow < n:
            arr3[maxRow] = finalVal
            maxRow += 1

        C[:, col] = arr3

        arr1, arr2, arr3 = arr2, arr3, arr1

    return C

def TransformChebInPlace1DCOO(coeffs, alpha, beta):
    n = coeffs.shape[0]

    C_dense = build_cheb_transform_matrix(n, alpha, beta)
    C_dense[np.abs(C_dense) <= 1e-16] = 0.0
    C = COO.from_numpy(C_dense)

    old_shape = coeffs.shape
    mat = coeffs.reshape((n, -1))

    return (matmul(C, mat)).reshape(old_shape)

def _precompute_C_columns(C_dense, tol=1e-16):
    """
    For each column j of C, store only the nonzero rows k and values C[k, j].
    """
    C_dense = C_dense.copy()
    C_dense[np.abs(C_dense) <= tol] = 0.0

    col_rows = []
    col_vals = []

    n = C_dense.shape[1]
    for j in range(n):
        rows = np.nonzero(C_dense[:, j])[0]
        vals = C_dense[rows, j]

        col_rows.append(rows)
        col_vals.append(vals)

    return col_rows, col_vals


def TransformChebInPlace1DCOO_manual(coeffs, alpha, beta, tol=1e-16):
    """
    Apply a 1D Chebyshev transform along axis 0 of a COO coefficient tensor.

    Computes:
        new_coeffs[k, ...] = sum_j C[k, j] * coeffs[j, ...]

    without using generic sparse matmul.
    """
    n = coeffs.shape[0]

    if (alpha == 1.0 and beta == 0.0) or n == 1:
        return coeffs

    C_dense = build_cheb_transform_matrix(n, alpha, beta)
    col_rows, col_vals = _precompute_C_columns(C_dense, tol=tol)

    coords = coeffs.coords
    data = coeffs.data
    ndim = coeffs.ndim

    # Count how many output terms we will create.
    out_nnz = 0
    for p in range(coeffs.nnz):
        old_j = coords[0, p]
        out_nnz += len(col_rows[old_j])

    new_coords = np.empty((ndim, out_nnz), dtype=coords.dtype)
    new_data = np.empty(out_nnz, dtype=data.dtype)

    out_idx = 0
    for p in range(coeffs.nnz):
        old_j = coords[0, p]
        old_val = data[p]

        rows = col_rows[old_j]
        vals = col_vals[old_j]

        for k, cval in zip(rows, vals):
            new_coords[:, out_idx] = coords[:, p]
            new_coords[0, out_idx] = k
            new_data[out_idx] = cval * old_val
            out_idx += 1

    result = COO(new_coords, new_data, shape=coeffs.shape)
    return result

@njit
def _precompute_C_columns_flat(C_dense, tol):
    """
    Convert dense C into a Numba-friendly compressed-column-like format.

    For each old Chebyshev index j, we store the nonzero entries of C[:, j].

    Column j entries are stored in:

        start = col_ptrs[j]
        end   = col_ptrs[j + 1]

        rows = col_rows[start:end]
        vals = col_vals[start:end]
    """
    n_rows, n_cols = C_dense.shape

    # First pass: count nonzeros in each column.
    counts = np.zeros(n_cols, dtype=np.int64)

    for j in range(n_cols):
        count = 0
        for i in range(n_rows):
            if abs(C_dense[i, j]) > tol:
                count += 1
        counts[j] = count

    # Build column pointer array.
    col_ptrs = np.empty(n_cols + 1, dtype=np.int64)
    col_ptrs[0] = 0

    for j in range(n_cols):
        col_ptrs[j + 1] = col_ptrs[j] + counts[j]

    total_nnz = col_ptrs[n_cols]

    col_rows = np.empty(total_nnz, dtype=np.int64)
    col_vals = np.empty(total_nnz, dtype=C_dense.dtype)

    # Second pass: fill row/value arrays.
    for j in range(n_cols):
        idx = col_ptrs[j]

        for i in range(n_rows):
            val = C_dense[i, j]

            if abs(val) > tol:
                col_rows[idx] = i
                col_vals[idx] = val
                idx += 1

    return col_ptrs, col_rows, col_vals


@njit
def _expand_cheb_axis0_coo(coords, data, ndim, col_ptrs, col_rows, col_vals):
    """
    Apply the Chebyshev transform along axis 0 by expanding COO coordinates.

    Computes:

        new_coeffs[k, ...] += C[k, j] * coeffs[j, ...]

    where j is coords[0, p] for each nonzero coefficient.
    """
    nnz = data.shape[0]

    # Count how many output terms we will create.
    out_nnz = 0

    for p in range(nnz):
        old_j = coords[0, p]
        out_nnz += col_ptrs[old_j + 1] - col_ptrs[old_j]

    new_coords = np.empty((ndim, out_nnz), dtype=coords.dtype)
    new_data = np.empty(out_nnz, dtype=data.dtype)

    out_idx = 0

    for p in range(nnz):
        old_j = coords[0, p]
        old_val = data[p]

        start = col_ptrs[old_j]
        end = col_ptrs[old_j + 1]

        for q in range(start, end):
            new_axis0 = col_rows[q]
            cval = col_vals[q]

            # Copy old coordinate.
            for d in range(ndim):
                new_coords[d, out_idx] = coords[d, p]

            # Replace axis 0 coordinate with transformed row index.
            new_coords[0, out_idx] = new_axis0

            # Multiply coefficient value.
            new_data[out_idx] = cval * old_val

            out_idx += 1

    return new_coords, new_data


@njit
def _TransformChebInPlace1DCOO_manual_helper(coords, data, shape0, ndim, alpha, beta, tol):
    """
    Numba helper for TransformChebInPlace1DCOO_manual.

    Builds dense C, compresses its columns, and expands the COO representation.
    """
    C_dense = build_cheb_transform_matrix(shape0, alpha, beta)

    col_ptrs, col_rows, col_vals = _precompute_C_columns_flat(C_dense, tol)

    new_coords, new_data = _expand_cheb_axis0_coo(
        coords,
        data,
        ndim,
        col_ptrs,
        col_rows,
        col_vals,
    )

    return new_coords, new_data


def TransformChebInPlace1DCOO_manual2(coeffs, alpha, beta, tol=1e-16):
    """
    Apply a 1D Chebyshev transform along axis 0 of a COO coefficient tensor.

    This avoids generic sparse matmul:

        mat = coeffs.reshape((n, -1))
        out = matmul(C, mat).reshape(old_shape)

    and instead directly expands COO coordinates using the nonzero columns of C.
    """
    n = coeffs.shape[0]

    if (alpha == 1.0 and beta == 0.0) or n == 1:
        return coeffs

    new_coords, new_data = _TransformChebInPlace1DCOO_manual_helper(
        coeffs.coords,
        coeffs.data,
        n,
        coeffs.ndim,
        alpha,
        beta,
        tol,
    )

    return COO(new_coords, new_data, shape=coeffs.shape)

def getConstantTermCOO(M):
    if M.nnz == 0:
        return 0.0

    candidates = np.flatnonzero(M.coords[0] == 0)

    if candidates.size == 0:
        return 0.0

    if M.coords.shape[0] > 1:
        candidates = candidates[np.all(M.coords[1:, candidates] == 0, axis=0)]

    if candidates.size == 0:
        return 0.0

    return np.sum(M.data[candidates])

@njit
def _getLinearTermsCOO_core(coords, data, shape):
    dim = coords.shape[0]
    nnz = data.shape[0]

    A = np.zeros(dim, dtype=data.dtype)

    for k in range(nnz):
        coord_sum = 0
        one_dim = -1

        for d in range(dim):
            idx = coords[d, k]

            if idx == 1:
                coord_sum += 1
                one_dim = d
            elif idx != 0:
                coord_sum = -1
                break

        if coord_sum == 1:
            # Matches original dense behavior:
            # if shape[d] == 1, linear coefficient is treated as 0.
            if shape[one_dim] > 1:
                A[one_dim] += data[k]

    return A

def getLinearTermsCOO(M):
    return _getLinearTermsCOO_core(
        M.coords,
        M.data,
        np.array(M.shape, dtype=np.int64),
    )

def trimOneCoo(M, allowedErrorIncrease, error):
    """
    Trim one sparse.COO Chebyshev coefficient tensor.

    Returns
    -------
    M : COO
        Trimmed COO coefficient tensor.

    allowedErrorIncrease : float
        Remaining allowed error increase.

    error : float
        Updated error bound.
    """

    dim = M.ndim

    for currDim in range(dim):
        # Keep trimming the current dimension while the highest-degree slice
        # is small enough and we preserve at least degree 2, i.e. shape >= 3.
        while M.shape[currDim] > 3:
            last_index = M.shape[currDim] - 1

            # Entries in the highest-degree slice of currDim.
            last_mask = M.coords[currDim] == last_index

            # Sum abs values of coefficients in that slice.
            if np.any(last_mask):
                lastSum = np.sum(np.abs(M.data[last_mask]))
            else:
                lastSum = 0.0

            if lastSum >= allowedErrorIncrease:
                break

            # Remove entries from the highest-degree slice.
            keep_mask = ~last_mask

            new_coords = M.coords[:, keep_mask]
            new_data = M.data[keep_mask]

            new_shape = list(M.shape)
            new_shape[currDim] -= 1
            new_shape = tuple(new_shape)

            M = COO(
                coords=new_coords,
                data=new_data,
                shape=new_shape,
            )

            allowedErrorIncrease -= lastSum
            error += lastSum

    return M, error

