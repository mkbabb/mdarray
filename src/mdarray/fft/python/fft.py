"""Mixed-radix self-sorting FFT with Bluestein's algorithm for prime lengths.

Implements Temperton's (1983) staged mixed-radix framework as the sole 1-D FFT
engine.  The DFT of length N = n_1 · n_2 · ... · n_k is computed as k stages,
each performing n_i-point butterflies across N/n_i groups, with twiddle factor
premultiplication from a precomputed flat table.

Hand-optimized radix-2, 3, 4, 5, and 7 butterflies (matching Temperton's
Table I operation counts) are dispatched for those factors.  Arbitrary prime
factors within a composite N use the generic O(n²) ``radixg`` butterfly—n is
small since it is just one factor in the decomposition.

When N itself is prime (and N > 4), Bluestein's chirp-z algorithm reduces the
problem to a power-of-2 convolution via ``_fft_staged``, achieving O(N log N).

The N-dimensional FFT uses **hypercube dimensional gliding**: the flat contiguous
buffer is treated as an N-dimensional hypercube, and 1-D FFTs are applied along
each axis by extracting fibers via stride arithmetic—no reshaping, no transposing,
no intermediate buffers.  The stride system that powers ``advance()`` (the
mixed-radix odometer for iteration) is the same system that enables fiber
extraction for the N-D transform.  This is the architectural through-line of the
library: the strided hypercube is the universal data structure, and dimensional
gliding is the universal traversal primitive.

References
----------
Temperton, C. "Self-Sorting Mixed-Radix Fast Fourier Transforms."
    J. Comput. Phys. 52, 1-23 (1983).
Bluestein, L.I. "A Linear Filtering Approach to the Computation of the
    Discrete Fourier Transform." IEEE Trans. Audio Electroacoust. 18(4),
    451-455 (1970).
"""

from __future__ import annotations

import cmath
import math

from ...array import mdarray
from ...core.helper import get_strides
from .butterflies import radix2, radix3, radix4, radix5, radix7, radixg
from .factorize import factorize
from .twiddle import twiddle_table

TWOPI = 2 * math.pi

# ---------------------------------------------------------------------------
# Butterfly dispatch
# ---------------------------------------------------------------------------

_BUTTERFLY = {2: radix2, 3: radix3, 4: radix4, 5: radix5, 7: radix7}


def _dispatch(n: int):
    """Return the butterfly function for factor *n*."""
    return _BUTTERFLY.get(n, radixg)


# ---------------------------------------------------------------------------
# Small-N base case
# ---------------------------------------------------------------------------


def _dft_naive(x: list[complex]) -> list[complex]:
    """Direct DFT computation, O(N²).  Used as base case for N ≤ 4."""
    N = len(x)
    X = [0j] * N
    for k in range(N):
        s = 0j
        for n in range(N):
            s += x[n] * cmath.exp(-2j * math.pi * k * n / N)
        X[k] = s
    return X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Primality test for FFT dispatch."""
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


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 ≥ *n*."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _digit_reverse(p: int, factors: list[int], N: int) -> int:
    """Compute the mixed-radix digit-reversal of position *p*.

    Decomposes *p* into mixed-radix digits under the factor list, then
    reconstructs the index with reversed digit weighting.
    """
    # Extract digits: p = d_0 * (N/f_0) + d_1 * (N/(f_0*f_1)) + ... + d_{L-1}
    remaining = p
    digits = []
    group_size = N
    for f in factors:
        group_size //= f
        digits.append(remaining // group_size)
        remaining %= group_size

    # Reconstruct with reversed weighting: k = d_0 + d_1*f_0 + d_2*f_0*f_1 + ...
    k = 0
    mul = 1
    for i, d in enumerate(digits):
        k += d * mul
        mul *= factors[i]

    return k


def _digit_reverse_permute(work: list[complex], factors: list[int], N: int) -> list[complex]:
    """Apply mixed-radix digit-reversal permutation.

    Returns a new list where ``result[digit_reverse(p)] = work[p]``.
    """
    result = [0j] * N
    for p in range(N):
        result[_digit_reverse(p, factors, N)] = work[p]
    return result


def _digit_reverse_input(work: list[complex], factors: list[int], N: int) -> list[complex]:
    """Digit-reverse the input: ``result[p] = work[digit_reverse(p)]``."""
    result = [0j] * N
    for p in range(N):
        result[p] = work[_digit_reverse(p, factors, N)]
    return result


# ---------------------------------------------------------------------------
# Temperton staged FFT
# ---------------------------------------------------------------------------


def _fft_staged(x: list[complex]) -> list[complex]:
    """Temperton's mixed-radix FFT on a flat complex list.

    Decimation-in-time (DIT) formulation with increasing stride.  The input is
    first rearranged via mixed-radix digit-reversal, then *k* butterfly stages
    are applied (one per prime factor), building up from small DFTs to the full
    transform.  The output is in natural order.

    At stage *s* with factor ``f_s`` and cumulative product
    ``pdiv = f_0 · ... · f_{s-1}``:

    1. ``pmul = f_s · pdiv`` is the current DFT size being assembled.
    2. ``groups = N / pmul`` independent groups are processed.
    3. Within each group, ``pdiv`` independent ``f_s``-point butterflies are
       performed, reading elements at stride ``pdiv``.

    The index for element *k* of the butterfly at group *m*, block *j* is::

        m * pmul + j + k * pdiv

    Each butterfly operates on a disjoint set of indices, so the computation
    is fully in-place.  The DIT butterflies premultiply inputs by twiddle
    factors from the precomputed table.

    Parameters
    ----------
    x : list[complex]
        Input sequence of length N (must be composite or 1).

    Returns
    -------
    list[complex]
        The DFT of *x*.
    """
    N = len(x)
    if N <= 1:
        return list(x)

    factors = factorize(N)
    twiddles = twiddle_table(factors)

    # Digit-reverse the input using the REVERSED factor list so that the
    # DIT increasing-stride butterflies read the correct elements.
    rev_factors = list(reversed(factors))
    work = _digit_reverse_input(x, rev_factors, N)

    pdiv = 1
    twix = 0

    for s in range(len(factors)):
        ifac = factors[s]
        butterfly = _dispatch(ifac)
        pmul = ifac * pdiv
        groups = N // pmul

        for m in range(groups):
            for j in range(pdiv):
                # Extract ifac elements at stride pdiv
                seq = [work[m * pmul + j + k * pdiv] for k in range(ifac)]

                # Apply butterfly with twiddle premultiplication.
                # pix=j for radixg's on-the-fly twiddle computation.
                butterfly(seq, twiddles, j, twix + j * (ifac - 1), pdiv)

                # Write back in-place (disjoint indices per butterfly)
                for k in range(ifac):
                    work[m * pmul + j + k * pdiv] = seq[k]

        twix += pdiv * (ifac - 1)
        pdiv *= ifac

    return work


# ---------------------------------------------------------------------------
# Bluestein's algorithm (chirp-z transform)
# ---------------------------------------------------------------------------


def _bluestein(x: list[complex]) -> list[complex]:
    """Bluestein's algorithm for prime-length DFT in O(N log N).

    Uses the chirp-z identity to reduce the DFT to a circular convolution:

        nk = -(k - n)²/2 + n²/2 + k²/2

    so that:

        X[k] = chirp[k] · Σ_n (x[n] · chirp[n]) · conj(chirp[k - n])

    The convolution is computed via FFT of a power-of-2 length M ≥ 2N - 1,
    which ``_fft_staged`` handles efficiently (all radix-2 butterflies).

    Phase reduction (``n² mod 2N``) keeps exponents small for numerical
    stability.

    Parameters
    ----------
    x : list[complex]
        Input sequence of prime length N.

    Returns
    -------
    list[complex]
        The DFT of *x*.
    """
    N = len(x)
    M = _next_power_of_2(2 * N - 1)

    # Build chirp: chirp[n] = exp(-iπ(n² mod 2N)/N)
    chirp = [0j] * N
    for n in range(N):
        nsq_mod = (n * n) % (2 * N)
        chirp[n] = cmath.exp(-1j * math.pi * nsq_mod / N)

    # Modulate input: a[n] = x[n] * chirp[n], zero-padded to M
    a = [0j] * M
    for n in range(N):
        a[n] = x[n] * chirp[n]

    # Build chirp kernel (symmetric wrap): b[n] = conj(chirp[n])
    b = [0j] * M
    b[0] = chirp[0].conjugate()
    for n in range(1, N):
        b[n] = chirp[n].conjugate()
        b[M - n] = chirp[n].conjugate()

    # Convolve via FFT: C = IFFT(FFT(a) · FFT(b))
    A = _fft_staged(a)
    B = _fft_staged(b)

    AB = [A[i] * B[i] for i in range(M)]

    # IFFT via conjugate trick: IFFT(X) = conj(FFT(conj(X))) / M
    AB_conj = [z.conjugate() for z in AB]
    c = _fft_staged(AB_conj)

    # Post-multiply by chirp and scale
    result = [0j] * N
    for k in range(N):
        result[k] = chirp[k] * c[k].conjugate() / M

    return result


# ---------------------------------------------------------------------------
# 1-D FFT dispatch
# ---------------------------------------------------------------------------


def _cfft_list(data: list[complex]) -> list[complex]:
    """Core 1-D FFT dispatch on a flat complex list.

    Decision flow::

        N ≤ 1            → identity
        N ≤ 4            → _dft_naive  (direct O(N²), cheap for tiny N)
        N prime, N > 4   → _bluestein  (chirp-z, O(N log N))
        N composite      → _fft_staged (Temperton butterflies + twiddles)
    """
    N = len(data)
    if N <= 1:
        return list(data)
    if N <= 4:
        return _dft_naive(data)
    if _is_prime(N):
        return _bluestein(data)
    return _fft_staged(data)


def cfft(arr: mdarray) -> mdarray:
    """Compute the 1-D discrete Fourier transform.

    Parameters
    ----------
    arr : mdarray
        Input array (treated as 1-D).

    Returns
    -------
    mdarray
        Complex DFT of the input.
    """
    size = arr.size
    if size == 0:
        return mdarray(shape=[0], data=[])
    if size == 1:
        return mdarray(shape=[1], data=[complex(arr.data[0])])

    data = [complex(x) for x in arr.data]
    result = _cfft_list(data)
    return mdarray(shape=[size], data=result)


# ---------------------------------------------------------------------------
# Inverse 1-D FFT
# ---------------------------------------------------------------------------


def ifft(arr: mdarray) -> mdarray:
    """Compute the 1-D inverse discrete Fourier transform.

    Uses the conjugate trick: IDFT(X) = conj(DFT(conj(X))) / N.
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


# ---------------------------------------------------------------------------
# N-D FFT — Hypercube Dimensional Gliding
# ---------------------------------------------------------------------------


def fftn(arr: mdarray, axes: list[int] | None = None) -> mdarray:
    """Compute the N-D discrete Fourier transform via dimensional gliding.

    The N-dimensional DFT is separable: it decomposes into independent 1-D DFTs
    along each axis.  **Hypercube dimensional gliding** exploits this by walking
    the flat buffer with stride arithmetic:

    1. Fix an axis *a* with size ``n_a`` and stride ``s_a``.
    2. Enumerate all **fibers** -- 1-D slices along axis *a* -- by iterating over
       all coordinates in the orthogonal (d-1)-dimensional hyperplane.
    3. For each fiber, compute the base offset from the orthogonal coordinates,
       then extract ``n_a`` elements at positions ``base + k * s_a`` for
       k ∈ [0, n_a).
    4. Apply the 1-D FFT (``_fft_staged`` or ``_bluestein``) to the fiber.
    5. Write results back to the same positions.
    6. Repeat for each axis.

    The algorithm requires no reshaping, transposing, or intermediate buffers.  The stride system
    provides O(1) random access into the hypercube along any axis—dimensional
    gliding is the traversal primitive.

    The mathematical guarantee: separability of the DFT kernel as a Kronecker
    product ``DFT_{s_0} ⊗ DFT_{s_1} ⊗ ... ⊗ DFT_{s_{d-1}}`` means independent
    1-D transforms along each axis.  Dimensional gliding is the implementation
    of this identity on the flat buffer.

    Parameters
    ----------
    arr : mdarray
        Input N-D array.
    axes : list[int] | None
        Axes along which to compute the FFT.  ``None`` means all axes.

    Returns
    -------
    mdarray
        Complex N-D DFT of the input, same shape as *arr*.
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

        # Orthogonal axes: all axes except the current one
        other_axes = [i for i in range(mdim) if i != axis]
        other_shapes = [shape[i] for i in other_axes]
        other_strides = [strides[i] for i in other_axes]

        # Glide over all fibers along this axis
        for fiber_idx in range(n_fibers):
            # Compute base offset from orthogonal coordinates
            base = 0
            remaining = fiber_idx
            for oi in range(len(other_axes) - 1, -1, -1):
                dim_size = other_shapes[oi]
                stride = other_strides[oi]
                if dim_size > 0:
                    coord = remaining % dim_size
                    remaining //= dim_size
                    base += coord * stride

            # Extract fiber: n_axis elements at stride axis_stride
            fiber = [data[base + k * axis_stride] for k in range(n_axis)]

            # Apply 1-D FFT
            fft_result = _cfft_list(fiber) if n_axis > 1 else list(fiber)

            # Write back
            for k in range(n_axis):
                data[base + k * axis_stride] = fft_result[k]

    return mdarray(shape=list(shape), data=data)


# ---------------------------------------------------------------------------
# Inverse N-D FFT
# ---------------------------------------------------------------------------


def ifftn(arr: mdarray, axes: list[int] | None = None) -> mdarray:
    """Compute the N-D inverse discrete Fourier transform.

    Conjugates the input, applies ``fftn``, conjugates and scales the output.
    Inherits the dimensional gliding algorithm from ``fftn``.
    """
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


# ---------------------------------------------------------------------------
# Real FFT
# ---------------------------------------------------------------------------


def rfft(arr: mdarray) -> mdarray:
    """Compute the 1-D FFT for real-valued input.

    Returns the non-redundant half (N//2 + 1 points).
    """
    result = cfft(arr)
    n_out = arr.size // 2 + 1
    return mdarray(shape=[n_out], data=result.data[:n_out])
