"""Tests for linear algebra operations."""

from __future__ import annotations


def test_diagonal_from_1d():
    from mdarray import tomdarray
    from mdarray.linalg import diagonal
    arr = tomdarray([5, 6, 7])
    result = diagonal(arr)
    assert result.shape == [3, 3]
    assert result.data[0] == 5
    # col_stride = 3, so diag indices: 0, 4, 8
    assert result.data[4] == 6
    assert result.data[8] == 7
    # Off-diagonals should be zero
    assert result.data[1] == 0
    assert result.data[3] == 0


def test_diagonal_from_2d():
    from mdarray import tomdarray
    from mdarray.linalg import diagonal
    arr = tomdarray([[5, 0, 0], [0, 6, 0], [0, 0, 7]])
    result = diagonal(arr)
    assert result.shape == [3]
    assert result.data == [5, 6, 7]


def test_identity():
    from mdarray.linalg import identity
    eye = identity(3)
    assert eye.shape == [3, 3]
    # Diagonal should be 1
    assert eye.data[0] == 1
    assert eye.data[4] == 1
    assert eye.data[8] == 1
    # Off-diagonal should be 0
    assert eye.data[1] == 0


def test_trace():
    from mdarray import tomdarray
    from mdarray.linalg import trace
    arr = tomdarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert trace(arr) == 15.0


def test_determinant_identity():
    from mdarray.linalg import determinant, identity
    eye = identity(3)
    assert abs(determinant(eye) - 1.0) < 1e-10


def test_determinant_known():
    from mdarray import tomdarray
    from mdarray.linalg import determinant
    # [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
    arr = tomdarray([[1, 2], [3, 4]])
    d = determinant(arr)
    assert abs(d - (-2.0)) < 1e-10


def test_gaussian_elim_rref():
    from mdarray import tomdarray
    from mdarray.linalg import gaussian_elim
    # Simple 2x2 matrix
    arr = tomdarray([[2, 4], [1, 3]])
    result = gaussian_elim(arr, rref=True)
    # Should produce identity-like RREF
    data = result.data
    # Check diagonal is 1 (or close to it)
    assert abs(data[0] - 1.0) < 1e-10
    assert abs(data[3] - 1.0) < 1e-10


def test_dot_2x2():
    """Test matrix multiplication against known result."""
    from mdarray import tomdarray
    from mdarray.linalg import dot

    A = tomdarray([[1, 0], [0, 1]])  # Identity
    B = tomdarray([[5, 6], [7, 8]])

    result = dot(A, B)
    # A * B should equal B for identity A
    assert result.data == B.data


def test_lu_decomposition():
    from mdarray import tomdarray
    from mdarray.linalg import lu

    arr = tomdarray([[2, 4], [1, 3]])
    L, U = lu(arr)

    # L should be lower triangular with unit diagonal
    assert abs(L.data[0] - 1.0) < 1e-10  # L[0,0]
    assert abs(L.data[3] - 1.0) < 1e-10  # L[1,1]

    # U should be upper triangular
    # U[1,0] should be 0
    assert abs(U.data[1 * U.strides[1]] - 0.0) < 1e-10


def test_qr_decomposition():
    from mdarray import tomdarray
    from mdarray.linalg import qr

    arr = tomdarray([[1, 0], [0, 1], [1, 1]])
    _Q, R = qr(arr)

    # R is [n, n] = [2, 2]. With strides [1, 2]:
    # R[0,0] = data[0], R[1,0] = data[1], R[0,1] = data[2], R[1,1] = data[3]
    # Upper triangular means R[1,0] = 0
    # In our layout with shape [2,2], R[col=1, row=0] = data[1] is below diagonal
    # Check R is approximately upper triangular
    assert R.shape == [2, 2]
    # The R matrix from QR should have the subdiagonal element ~0
    # R[1,0] in col-major-like layout: data[1*1 + 0*strides[1]] = data[1]
    # But our QR stores R_data[j + i * n], so R[row=1,col=0] = R_data[0 + 1*2] = data[2]
    assert abs(R.data[2]) < 1e-10  # R[row=1, col=0]


def test_norm():
    from mdarray import tomdarray
    from mdarray.linalg import norm

    arr = tomdarray([3, 4])
    assert abs(norm(arr) - 5.0) < 1e-10
