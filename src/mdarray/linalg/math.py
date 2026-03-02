"""Linear algebra operations.

References:
    Golub, G.H. and Van Loan, C.F. Matrix Computations. 4th ed., Johns Hopkins, 2013.
    Trefethen, L.N. and Bau, D. Numerical Linear Algebra. SIAM, 1997.
"""

from __future__ import annotations

import math

from ..array import mdarray, zeros
from ..core.helper import swap

__all__ = [
    "determinant",
    "dot",
    "gaussian_elim",
    "inverse",
    "lu",
    "norm",
    "qr",
    "solve",
    "trace",
]


def dot(arr1: mdarray, arr2: mdarray) -> mdarray:
    """Matrix multiplication (dot product) for 2-D arrays.

    For arr1 of shape [M, K] and arr2 of shape [K, N], produces output of shape [M, N].
    """
    shape1 = arr1.shape
    shape2 = arr2.shape

    if arr1.mdim != 2 or arr2.mdim != 2:
        raise ValueError("dot requires 2-D arrays")

    M = shape1[1]
    K1 = shape1[0]
    K2 = shape2[1]
    N = shape2[0]

    stride1_col = arr1.strides[0]
    stride1_row = arr1.strides[1]
    stride2_col = arr2.strides[0]
    stride2_row = arr2.strides[1]

    if K1 != K2:
        raise ValueError(f"Inner dimensions must match for dot product: {K1} != {K2}")

    arr_out = zeros([N, M])
    out_stride_col = arr_out.strides[0]
    out_stride_row = arr_out.strides[1]

    for i in range(M):
        for j in range(N):
            s = 0
            for k in range(K1):
                a = arr1.data[k * stride1_col + i * stride1_row]
                b = arr2.data[j * stride2_col + k * stride2_row]
                s += a * b
            arr_out.data[j * out_stride_col + i * out_stride_row] = s

    return arr_out


def norm(arr: mdarray, metric: int = 2) -> float:
    """Compute the L-p norm of a 1-D array."""
    s = 0
    for i in range(arr.size):
        s += abs(arr.data[i]) ** metric
    return s ** (1 / metric)


def gaussian_elim(arr: mdarray, rref: bool = True) -> mdarray:
    """Gaussian elimination with partial pivoting.

    Operates in-place on arr. If rref=True, performs back-substitution
    to produce reduced row echelon form.
    """
    shape = arr.shape
    strides = arr.strides
    col_stride = strides[1]

    row = shape[1]
    col = shape[0]

    data = arr.data

    for i in range(min(row, col)):
        diag = i * (col_stride + 1)
        pix = i
        pivot = data[diag]

        for j in range(i + 1, row):
            ppix_base = j * col_stride
            ppivot = data[ppix_base + i]

            if abs(ppivot) > abs(pivot):
                pix = j
                pivot = ppivot

        if pix != i:
            for j in range(col):
                swap(data, pix * col_stride + j, i * col_stride + j)

        if pivot != 0:
            for j in range(i + 1, row):
                ppix_base = j * col_stride
                ppivot = data[ppix_base + i]

                mul = -ppivot / pivot
                data[ppix_base + i] = 0
                for k in range(i + 1, col):
                    data[ppix_base + k] += mul * data[i * col_stride + k]

    if rref:
        for i in range(min(row, col) - 1, -1, -1):
            diag = i * (col_stride + 1)
            pivot = data[diag]

            if pivot != 0:
                for k in range(col):
                    data[i * col_stride + k] /= pivot

                for j in range(i - 1, -1, -1):
                    factor = data[j * col_stride + i]
                    if factor != 0:
                        data[j * col_stride + i] = 0
                        for k in range(i + 1, col):
                            data[j * col_stride + k] -= factor * data[i * col_stride + k]

    return arr


def determinant(arr: mdarray) -> float:
    """Compute determinant via Gaussian elimination."""
    if arr.mdim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Determinant requires a square 2-D array")

    n = arr.shape[0]
    work = mdarray(shape=list(arr.shape), data=list(arr.data))

    strides = work.strides
    col_stride = strides[1]
    data = work.data
    det = 1.0
    swaps = 0

    for i in range(n):
        diag = i * (col_stride + 1)
        pix = i
        pivot = data[diag]

        for j in range(i + 1, n):
            ppivot = data[j * col_stride + i]
            if abs(ppivot) > abs(pivot):
                pix = j
                pivot = ppivot

        if pix != i:
            for k in range(n):
                swap(data, pix * col_stride + k, i * col_stride + k)
            swaps += 1

        if pivot == 0:
            return 0.0

        det *= pivot

        for j in range(i + 1, n):
            mul = data[j * col_stride + i] / pivot
            data[j * col_stride + i] = 0
            for k in range(i + 1, n):
                data[j * col_stride + k] -= mul * data[i * col_stride + k]

    return det * ((-1) ** swaps)


def solve(A: mdarray, b: mdarray) -> mdarray:
    """Solve Ax = b via Gaussian elimination with back-substitution."""
    if A.mdim != 2:
        raise ValueError("A must be 2-D")

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square")

    aug_data = [0.0] * (n * (n + 1))
    col_stride_a = A.strides[1]
    col_stride_aug = n + 1

    for i in range(n):
        for j in range(n):
            aug_data[j + i * col_stride_aug] = A.data[j + i * col_stride_a]
        aug_data[n + i * col_stride_aug] = b.data[i]

    aug = mdarray(shape=[n + 1, n], data=aug_data)
    gaussian_elim(aug, rref=True)

    result = zeros([n])
    for i in range(n):
        result.data[i] = aug.data[n + i * col_stride_aug]

    return result


def inverse(arr: mdarray) -> mdarray:
    """Compute matrix inverse via Gauss-Jordan on augmented [A|I]."""
    if arr.mdim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Inverse requires a square 2-D array")

    n = arr.shape[0]

    aug_cols = 2 * n
    aug_data = [0.0] * (aug_cols * n)
    col_stride_a = arr.strides[1]

    for i in range(n):
        for j in range(n):
            aug_data[j + i * aug_cols] = arr.data[j + i * col_stride_a]
        aug_data[n + i + i * aug_cols] = 1.0

    aug = mdarray(shape=[aug_cols, n], data=aug_data)
    gaussian_elim(aug, rref=True)

    result = zeros([n, n])
    col_stride_r = result.strides[1]
    for i in range(n):
        for j in range(n):
            result.data[j + i * col_stride_r] = aug.data[n + j + i * aug_cols]

    return result


def lu(arr: mdarray) -> tuple[mdarray, mdarray]:
    """LU decomposition via Gaussian elimination.

    Returns (L, U) where A = L * U, L is lower triangular with unit diagonal,
    U is upper triangular.
    """
    if arr.mdim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("LU requires a square 2-D array")

    n = arr.shape[0]

    L = zeros([n, n])
    U = mdarray(shape=[n, n], data=list(arr.data))

    col_stride = U.strides[1]
    L_stride = L.strides[1]

    for i in range(n):
        L.data[i + i * L_stride] = 1.0

    for i in range(n):
        pivot = U.data[i * (col_stride + 1)]
        if pivot == 0:
            continue

        for j in range(i + 1, n):
            mul = U.data[i + j * col_stride] / pivot
            L.data[i + j * L_stride] = mul
            U.data[i + j * col_stride] = 0
            for k in range(i + 1, n):
                U.data[k + j * col_stride] -= mul * U.data[k + i * col_stride]

    return L, U


def qr(arr: mdarray) -> tuple[mdarray, mdarray]:
    """QR decomposition via modified Gram-Schmidt.

    Returns (Q, R) where A = Q * R, Q is orthogonal, R is upper triangular.
    """
    if arr.mdim != 2:
        raise ValueError("QR requires a 2-D array")

    m = arr.shape[1]
    n = arr.shape[0]

    col_stride = arr.strides[1]

    cols: list[list[float]] = []
    for j in range(n):
        col: list[float] = []
        for i in range(m):
            col.append(arr.data[j + i * col_stride])
        cols.append(col)

    Q_cols: list[list[float]] = []
    R_data = [0.0] * (n * n)

    for j in range(n):
        v = list(cols[j])

        for i in range(len(Q_cols)):
            r = sum(Q_cols[i][k] * v[k] for k in range(m))
            R_data[j + i * n] = r
            for k in range(m):
                v[k] -= r * Q_cols[i][k]

        v_norm = math.sqrt(sum(x * x for x in v))
        R_data[j + j * n] = v_norm

        if v_norm > 1e-15:
            Q_cols.append([x / v_norm for x in v])
        else:
            Q_cols.append([0.0] * m)

    Q = zeros([n, m])
    Q_stride = Q.strides[1]
    for j in range(len(Q_cols)):
        for i in range(m):
            Q.data[j + i * Q_stride] = Q_cols[j][i]

    R = mdarray(shape=[n, n], data=R_data)
    return Q, R


def trace(arr: mdarray) -> float:
    """Sum of diagonal elements."""
    if arr.mdim != 2:
        raise ValueError("trace requires a 2-D array")

    n = min(arr.shape[0], arr.shape[1])
    col_stride = arr.strides[1]
    s = 0.0
    for i in range(n):
        s += arr.data[i * (col_stride + 1)]
    return s
