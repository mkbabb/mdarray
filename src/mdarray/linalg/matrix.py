from __future__ import annotations

from ..array import mdarray, ones, zeros

__all__ = ["diagonal", "identity"]


def diagonal(arr: mdarray) -> mdarray:
    """Extract diagonal from 2-D array or construct diagonal matrix from 1-D array."""
    mdim = arr.mdim
    size = arr.size
    shape = arr.shape

    if mdim == 1:
        arr_out = zeros([size, size])
        col_stride = arr_out.strides[1]
        for i in range(size):
            arr_out.data[i * (col_stride + 1)] = arr.data[i]
    elif mdim == 2:
        n = min(shape[0], shape[1])
        arr_out = zeros([n])
        col_stride = arr.strides[1]
        for i in range(n):
            arr_out.data[i] = arr.data[i * (col_stride + 1)]
    else:
        raise ValueError(f"diagonal requires 1-D or 2-D array, got {mdim}-D")

    return arr_out


def identity(order: int = 2) -> mdarray:
    """Create an identity matrix of the given order."""
    return diagonal(ones([order]))
