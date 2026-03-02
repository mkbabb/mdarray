"""Mixed-radix self-sorting FFT.

Implements the Cooley-Tukey mixed-radix FFT using Temperton's (1983) matrix
factorization framework. This implementation operates on flat Python lists
for reliability, with the mdarray interface wrapping the computation.

The DFT of length N = n_1 * n_2 * ... * n_k is computed as k stages,
each performing n_i-point butterflies across N/n_i groups, with twiddle
factor premultiplication.

References:
    Temperton, C. "Self-Sorting Mixed-Radix Fast Fourier Transforms."
    J. Comput. Phys. 52, 1-23 (1983).
"""

from __future__ import annotations

import cmath
import math

from ...array import mdarray
from ...core.helper import get_strides
from .factorize import factorize

TWOPI = 2 * math.pi


def _dft_naive(x: list[complex]) -> list[complex]:
    """Direct DFT computation, O(N^2). Used for small/prime sizes."""
    N = len(x)
    X = [0j] * N
    for k in range(N):
        s = 0j
        for n in range(N):
            s += x[n] * cmath.exp(-2j * math.pi * k * n / N)
        X[k] = s
    return X


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _fft_cooley_tukey(x: list[complex]) -> list[complex]:
    """Mixed-radix Cooley-Tukey FFT on a flat complex list.

    For simplicity and correctness, we use a standard approach:
    decompose into factors, apply the Cooley-Tukey butterfly per stage.
    """
    N = len(x)
    if N <= 1:
        return list(x)
    if N <= 4:
        return _dft_naive(x)

    factors = factorize(N)

    if len(factors) == 1:
        return _dft_naive(x)

    N1 = factors[0]
    N2 = N // N1

    # Column FFTs (N1 columns, each of length N2)
    cols: list[list[complex]] = []
    for n1 in range(N1):
        col = [x[n1 + n2 * N1] for n2 in range(N2)]
        if N2 > 1:
            col = _fft_cooley_tukey(col)
        cols.append(col)

    # Twiddle and row FFTs
    result = [0j] * N
    for k2 in range(N2):
        row = [0j] * N1
        for n1 in range(N1):
            tw = cmath.exp(-2j * math.pi * k2 * n1 / N)
            row[n1] = cols[n1][k2] * tw

        if N1 <= 1:
            row_fft = row
        elif N1 <= 4 or _is_prime(N1):
            row_fft = _dft_naive(row)
        else:
            row_fft = _fft_cooley_tukey(row)

        for k1 in range(N1):
            result[k1 * N2 + k2] = row_fft[k1]

    return result


def cfft(arr: mdarray) -> mdarray:
    """Compute the 1-D discrete Fourier transform."""
    size = arr.size
    if size == 0:
        return mdarray(shape=[0], data=[])
    if size == 1:
        return mdarray(shape=[1], data=[complex(arr.data[0])])

    data = [complex(x) for x in arr.data]
    result = _fft_cooley_tukey(data)
    return mdarray(shape=[size], data=result)


def ifft(arr: mdarray) -> mdarray:
    """Compute the 1-D inverse discrete Fourier transform.

    Uses: IDFT(X) = conj(DFT(conj(X))) / N
    """
    size = arr.size
    if size == 0:
        return mdarray(shape=[0], data=[])
    if size == 1:
        return mdarray(shape=[1], data=[complex(arr.data[0])])

    conj_data = [complex(x).conjugate() for x in arr.data]
    conj_arr = mdarray(shape=[size], data=conj_data)

    result = cfft(conj_arr)

    for i in range(size):
        result._data[i] = result._data[i].conjugate() / size

    return result


def fftn(arr: mdarray, axes: list[int] | None = None) -> mdarray:
    """Compute the N-D discrete Fourier transform.

    Applies 1-D FFT along each axis in sequence.
    """
    shape = list(arr.shape)
    mdim = len(shape)

    if axes is None:
        axes = list(range(mdim))

    data = [complex(x) for x in arr.data]

    for axis in axes:
        strides = get_strides(shape)
        n_axis = shape[axis]
        axis_stride = strides[axis]

        total = 1
        for s in shape:
            total *= s
        n_fibers = total // n_axis

        other_axes = [i for i in range(mdim) if i != axis]
        other_shapes = [shape[i] for i in other_axes]
        other_strides = [strides[i] for i in other_axes]

        for fiber_idx in range(n_fibers):
            base = 0
            remaining = fiber_idx
            for oi in range(len(other_axes) - 1, -1, -1):
                dim_size = other_shapes[oi]
                stride = other_strides[oi]
                if dim_size > 0:
                    coord = remaining % dim_size
                    remaining //= dim_size
                    base += coord * stride

            fiber = [data[base + k * axis_stride] for k in range(n_axis)]

            fft_result = _fft_cooley_tukey(fiber) if n_axis > 1 else list(fiber)

            for k in range(n_axis):
                data[base + k * axis_stride] = fft_result[k]

    return mdarray(shape=list(shape), data=data)


def ifftn(arr: mdarray, axes: list[int] | None = None) -> mdarray:
    """Compute the N-D inverse discrete Fourier transform."""
    shape = list(arr.shape)
    if axes is None:
        axes = list(range(len(shape)))

    data = [complex(x).conjugate() for x in arr.data]
    conj_arr = mdarray(shape=list(shape), data=data)

    result = fftn(conj_arr, axes)

    total = 1
    for ax in axes:
        total *= shape[ax]
    for i in range(result.size):
        result._data[i] = result._data[i].conjugate() / total

    return result


def rfft(arr: mdarray) -> mdarray:
    """Compute the 1-D FFT for real-valued input.

    Returns the non-redundant half (N//2 + 1 points).
    """
    result = cfft(arr)
    n_out = arr.size // 2 + 1
    return mdarray(shape=[n_out], data=result.data[:n_out])
