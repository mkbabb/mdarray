# FFT

## Background

The FFT module descends from `fftplusplus` (May 2018), a C++ port of Swarztrauber's Fortran-77 FFTPACK with CPython bindings. That project went through three pure-Python iterations (`fftplusplus.py` through `fftplusplus_py3.py`), each refining the mixed-radix decomposition. The final Python iteration supported radix-2, 3, 4, 5, 7, and 11 with a generic fallback, but was 1-D only.

mdarray's FFT (January 2019) carried the algorithm forward and generalized it to N dimensions via stride-based fiber extraction. The current implementation operates on flat Python lists for the inner computation, with `mdarray` wrapping the interface.

## The discrete Fourier transform

For a sequence x of length N:

```
X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n / N)
```

Direct computation costs O(N^2). The FFT computes it in O(N log N) by exploiting the factorization of N.

## Mixed-radix Cooley-Tukey

When N = N1 * N2, the 1-D DFT can be rewritten as a 2-D computation:

1. Arrange x into an N2 x N1 matrix (columns of length N2).
2. Apply N1-point FFTs along each column.
3. Multiply by twiddle factors: `W_N^{k2 * n1}`.
4. Apply N2-point FFTs along each row.
5. Read out in transposed order.

This is Temperton's matrix factorization (1983, Eq. 4–12). The DFT matrix W_N for N = n_1 * n_2 * ... * n_k factors as:

```
W_N = T_k T_{k-1} ... T_2 T_1
```

where each stage T_i consists of small DFT butterflies (W_{n_i} tensored with identity), twiddle factor premultiplication (the diagonal D^{n_i}_{m_i}), and a permutation (P^{n_i}_{m_i}). The "self-sorting" property—output in natural order without a separate bit-reversal pass—comes from the permutation matrices being absorbed into the indexing scheme. The cost is two arrays (ping-pong buffers).

For N = n_1 * n_2 * ... * n_k, the process recurses through each factor, yielding O(N * sum(n_i)) operations total.

## Implementation

`fft/python/fft.py` implements this as `_fft_cooley_tukey()`:

```python
def _fft_cooley_tukey(x):
    N = len(x)
    factors = factorize(N)
    N1 = factors[0]
    N2 = N // N1

    # Column FFTs (recursive)
    cols = []
    for n1 in range(N1):
        col = [x[n1 + n2 * N1] for n2 in range(N2)]
        col = _fft_cooley_tukey(col)
        cols.append(col)

    # Twiddle multiplication + row FFTs
    result = [0j] * N
    for k2 in range(N2):
        row = [cols[n1][k2] * exp(-2j*pi*k2*n1/N) for n1 in range(N1)]
        row_fft = _fft_cooley_tukey(row)
        for k1 in range(N1):
            result[k1 * N2 + k2] = row_fft[k1]

    return result
```

Prime-length transforms fall back to `_dft_naive()`—O(N^2) direct computation. Correct for all N but slow for large primes.

## Factorization

`factorize(N)` returns the prime factorization ordered to prefer small factors first (2, 3, 5, 7, then remaining primes). Small factors give efficient butterflies; the ordering minimizes the operation count per Temperton's analysis.

## Twiddle factors

The twiddle factors are the diagonal matrices D^{n_i}_{m_i} in Temperton's factorization (Eq. 12). They encode the complex exponentials needed between butterfly stages:

```
tw[j, k] = exp(-2*pi*i * j * k / (n_i * pdiv))
```

where pdiv is the cumulative product of previous factors. `twiddle.py` precomputes these into a flat list indexed by stage.

## N-dimensional FFT

Temperton's Section 5 gives the framework: for multidimensional transforms, apply 1-D FFTs along each axis in sequence. The stride system makes this natural—different axes correspond to different strides through the flat data.

`fftn(arr, axes)` iterates over each axis, extracts fibers (1-D slices along that axis) using stride arithmetic, applies `_fft_cooley_tukey` to each fiber, and writes results back. The fiber extraction reduces to computing a base offset from the coordinates along all other axes, then stepping by `axis_stride`.

## Inverse FFT

`ifft(x) = conj(fft(conj(x))) / N`. Conjugate the input, forward-transform, conjugate the output, scale by 1/N. This reuses the forward FFT without duplicating the algorithm.

## Codelet generator

`fft/codegen/genfft.py` generates optimized straight-line code for fixed-radix butterflies. For a given radix n, it constructs the n x n DFT matrix symbolically, identifies trivial twiddle factors (w^0 = 1, w^{n/2} = -1, w^{n/4} = -i), applies algebraic simplifications and common subexpression elimination, and emits Python source with multiplications replaced by negations or real/imag swaps where possible.

This follows the same principle as FFTW's `genfft` (Frigo and Johnson, 2005): generated codelets for small transforms, generic code for everything else. The hand-optimized butterflies for radices 2–7 remain—they match Temperton's Table I operation counts—and the generator covers higher radices.

## References

- Temperton, C. "Self-Sorting Mixed-Radix Fast Fourier Transforms." *J. Comput. Phys.* **52**, 1–23 (1983).
- Cooley, J.W. and Tukey, J.W. "An Algorithm for the Machine Calculation of Complex Fourier Series." *Math. Comp.* **19**(90), 297–301 (1965).
- Gentleman, W.M. and Sande, G. "Fast Fourier Transforms—For Fun and Profit." *Proc. AFIPS Fall Joint Computer Conference* **29**, 563–578 (1966).
- Winograd, S. "On Computing the Discrete Fourier Transform." *Math. Comp.* **32**, 175–199 (1978).
- Van Loan, C.F. *Computational Frameworks for the Fast Fourier Transform.* SIAM, 1992.
- Frigo, M. and Johnson, S.G. "The Design and Implementation of FFTW3." *Proc. IEEE* **93**(2), 216–231 (2005).
- Oppenheim, A.V. and Schafer, R.W. *Digital Signal Processing.* Prentice-Hall, 1975.
