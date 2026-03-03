# FFT

## Background

The FFT module descends from `fftplusplus` (May 2018), a C++ port of Swarztrauber's Fortran-77 FFTPACK with CPython bindings. That project went through three pure-Python iterations (`fftplusplus.py` through `fftplusplus_py3.py`), each refining the mixed-radix decomposition. The final Python iteration supported radix-2, 3, 4, 5, 7, and 11 with a generic fallback, but was 1-D only.

mdarray's FFT (January 2019) carried the algorithm forward and generalized it to N dimensions via stride-based fiber extraction. The current implementation operates on flat Python lists for the inner computation, with `mdarray` wrapping the interface.

## The discrete Fourier transform

For a sequence x of length N:

```
X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n / N),    k = 0, 1, ..., N-1
```

Direct computation costs O(N^2). The FFT computes it in O(N log N) by exploiting the factorization of N.

## The strided hypercube

The central abstraction of the library. An N-dimensional array of shape `[s_0, s_1, ..., s_{d-1}]` is stored as a **flat contiguous buffer**. The **stride vector** `[sigma_0, sigma_1, ..., sigma_{d-1}]` maps multi-indices to flat offsets:

```
offset(i_0, i_1, ..., i_{d-1}) = sum_k  i_k * sigma_k
```

For row-major (C-order) layout, `sigma_0 = 1` and `sigma_k = s_0 * s_1 * ... * s_{k-1}`. This is computed by `get_strides()`.

The same stride system powers:

- **Element access**: `arr.data[offset]` via the stride formula.
- **Iteration**: `advance()` is a mixed-radix odometer with carry propagation, incrementing axis counters in stride units.
- **Broadcasting**: repeat counters replay elements along size-1 axes.
- **Fiber extraction**: selecting a 1-D slice along any axis by fixing the orthogonal coordinates and stepping by that axis's stride.

This last capability--fiber extraction via stride arithmetic--is what enables the N-dimensional FFT.

## Dimensional gliding

The N-dimensional DFT is **separable**: it decomposes as a Kronecker product of 1-D transforms:

```
DFT_{s_0 x s_1 x ... x s_{d-1}} = DFT_{s_0} (x) DFT_{s_1} (x) ... (x) DFT_{s_{d-1}}
```

where `(x)` denotes the Kronecker product. This means independent 1-D DFTs along each axis produce the full N-D result.

**Hypercube dimensional gliding** is the algorithm that implements this identity on the flat buffer:

1. Fix an axis `a` with size `n_a` and stride `sigma_a`.
2. Enumerate all **fibers**--1-D slices along axis `a`--by iterating over all coordinates in the orthogonal (d-1)-dimensional hyperplane.
3. For each fiber, compute the base offset from the orthogonal coordinates, then extract `n_a` elements at positions `base + k * sigma_a` for k in [0, n_a).
4. Apply the 1-D FFT to the fiber.
5. Write results back to the same positions.
6. Repeat for each axis.

The algorithm requires no reshaping, transposing, or intermediate buffers. The stride system provides O(1) random access into the hypercube along any axis--dimensional gliding is the traversal primitive. The `advance()` odometer and the fiber extraction loop are two faces of the same coin: both navigate the flat buffer via stride arithmetic on the hypercube.

## Mixed-radix factorization

When N = N1 * N2, a 1-D index n can be decomposed as:

```
n = n1 + n2 * N1       (n1 in [0, N1),  n2 in [0, N2))
k = k1 * N2 + k2       (k1 in [0, N1),  k2 in [0, N2))
```

Substituting into the DFT definition and separating the exponential:

```
X[k1*N2 + k2] = sum_{n2} [ sum_{n1} x[n1 + n2*N1] * W_{N1}^{n1*k1} ]
                * W_N^{n1*k2} * W_{N2}^{n2*k2}
```

This decomposes the length-N DFT into:

1. **Column FFTs**: N2 independent length-N1 DFTs (inner sum).
2. **Twiddle multiplication**: multiply by W_N^{n1*k2} (the diagonal matrix).
3. **Row FFTs**: N1 independent length-N2 DFTs (outer sum).

For N = n_1 * n_2 * ... * n_k, the process recurses through each factor, yielding a total operation count of O(N * sum(n_i)).

## Kronecker product formulation

Following Van Loan (1992), the DFT of length N = N1 * N2 factors as:

```
DFT_N = (DFT_{N1} (x) I_{N2}) * T * (I_{N1} (x) DFT_{N2}) * L
```

where:

- `L` is the stride permutation (perfect shuffle) that rearranges elements between the column and row phases.
- `T` is the twiddle diagonal: `T = diag(W_N^{j*k})` for the appropriate index pairs.
- `(x)` denotes the Kronecker product.
- `I_m` is the m x m identity matrix.

The left factor applies N2-point DFTs, the right factor applies N1-point DFTs, and the twiddle diagonal mediates between stages.

## Self-sorting property

Temperton's (1983) contribution is a **self-sorting** formulation: the permutation matrices L are absorbed into the indexing scheme, so the output appears in natural order without a separate bit-reversal (or digit-reversal) pass. The cost is two arrays (ping-pong buffers) that swap roles at each stage.

In our implementation, we use a decimation-in-time (DIT) formulation with increasing stride. The input is pre-permuted via mixed-radix digit-reversal, then k butterfly stages are applied (one per prime factor), building up from small DFTs to the full transform. The output lands in natural order.

At stage s with factor f_s and cumulative product pdiv = f_0 * ... * f_{s-1}:

1. `pmul = f_s * pdiv` is the current DFT size being assembled.
2. `groups = N / pmul` independent groups are processed.
3. Within each group, pdiv independent f_s-point butterflies read elements at stride pdiv.

The index for element k of the butterfly at group m, block j is:

```
m * pmul + j + k * pdiv
```

Each butterfly operates on a disjoint set of indices, so the computation is fully in-place.

## Twiddle factors

The twiddle factors are the diagonal matrices D in Temperton's factorization (Eq. 12). They encode the complex exponentials needed between butterfly stages:

```
tw[j, k] = exp(-2*pi*i * j * k / (f_s * pdiv))
```

where j indexes the group element (0 to pdiv-1), k indexes the butterfly input (1 to f_s-1), and pdiv is the cumulative product of previous factors.

`twiddle_table(ifax)` precomputes these into a flat list, laid out by stage, then by group element, then by twiddle index:

- For stage s with factor f_s and cumulative product pdiv:
  - There are pdiv groups, each needing (f_s - 1) twiddle factors.
  - Total entries for stage s: pdiv * (f_s - 1).
- The offset into the flat table for stage s is the sum of entries from all previous stages.

At j=0 (the first group element), all twiddles are 1, so the butterfly reduces to an untwiddled DFT.

## Optimized butterflies

Hand-optimized radix-2, 3, 4, 5, and 7 butterflies match Temperton's Table I operation counts:

| Radix | Real multiplications | Real additions |
|-------|---------------------|----------------|
| 2     | 0                   | 4              |
| 3     | 4                   | 12             |
| 4     | 0                   | 16             |
| 5     | 12                  | 24             |
| 7     | 36                  | 48             |

The algebraic structure exploited per radix:

- **Radix 2**: No multiplications needed--just add/subtract.
- **Radix 3**: Uses cos(2pi/3) = -1/2, sin(2pi/3) = sqrt(3)/2.
- **Radix 4**: Powers of i (the 4th root of unity) are {1, -i, -1, i}, eliminating all real multiplications.
- **Radix 5**: Precomputed sin/cos constants for 5th roots of unity, with careful grouping of symmetric/antisymmetric terms.
- **Radix 7**: Same symmetric structure, using precomputed cos(2k*pi/7) and sin(2k*pi/7) for k=1,2,3.

The generic `radixg` handles arbitrary prime factors within a composite N via O(n^2) direct DFT. Since n is small (just one factor in the decomposition), the quadratic cost is acceptable.

## Bluestein's algorithm

When N itself is prime (and > 4), the staged algorithm cannot decompose it. Bluestein's chirp-z algorithm reduces the problem to a circular convolution, achieving O(N log N) even for prime N.

The identity:

```
n*k = -(k - n)^2 / 2  +  n^2 / 2  +  k^2 / 2
```

transforms the DFT sum into:

```
X[k] = chirp[k] * sum_n  (x[n] * chirp[n]) * conj(chirp[k - n])
```

where `chirp[n] = exp(-i*pi*n^2/N)`. The inner sum is a **convolution** of the modulated input with the conjugate chirp kernel. This convolution is computed via FFT of a power-of-2 length M >= 2N - 1:

1. Modulate: `a[n] = x[n] * chirp[n]`, zero-pad to length M.
2. Build symmetric kernel: `b[0] = conj(chirp[0])`, `b[n] = b[M-n] = conj(chirp[n])`.
3. Convolve: `c = IFFT(FFT(a) * FFT(b))`.
4. Post-multiply: `X[k] = chirp[k] * c[k]`.

Since M is a power of 2, `_fft_staged` handles the inner FFTs efficiently (all radix-2 butterflies). Phase reduction (`n^2 mod 2N`) keeps exponents small for numerical stability.

**Mutual recursion note**: `_bluestein` calls `_fft_staged` on power-of-2 lengths (always composite). `_fft_staged` dispatches to `radixg` for prime factors *within* a composite N--it never calls `_bluestein`. No circular dependency.

## Codelet generator

`fft/codegen/genfft.py` generates optimized straight-line code for fixed-radix butterflies. For a given radix n, it:

1. Constructs the n x n DFT matrix symbolically.
2. Identifies trivial twiddle factors (w^0 = 1, w^{n/2} = -1, w^{n/4} = -i, w^{3n/4} = i).
3. Replaces multiplications with negations or real/imag swaps where possible.
4. Emits Python or Rust source code.

This follows the same principle as FFTW's `genfft` (Frigo and Johnson, 2005): generated codelets for small transforms, generic code for everything else. The hand-optimized butterflies for radices 2-7 remain--they match Temperton's Table I operation counts--and the generator covers higher radices.

## Factorization strategy

`factorize(N)` returns the prime factorization ordered to prefer small factors first (2, 3, 5, 7, then remaining primes). Small factors give efficient butterflies with lower operation counts per element. The ordering minimizes total operations per Temperton's analysis.

## Implementation architecture

The 1-D FFT dispatch:

```
N <= 1            -> identity
N <= 4            -> _dft_naive  (direct O(N^2), cheap for tiny N)
N prime, N > 4    -> _bluestein  (chirp-z, O(N log N))
N composite       -> _fft_staged (Temperton butterflies + precomputed twiddles)
```

The N-D FFT (`fftn`) applies dimensional gliding: for each axis, enumerate all fibers via stride arithmetic, apply `_cfft_list` to each fiber, write back. The inverse (`ifftn`) uses the conjugate trick: `IFFT(X) = conj(FFT(conj(X))) / N`.

## References

- Temperton, C. "Self-Sorting Mixed-Radix Fast Fourier Transforms." *J. Comput. Phys.* **52**, 1-23 (1983).
- Cooley, J.W. and Tukey, J.W. "An Algorithm for the Machine Calculation of Complex Fourier Series." *Math. Comp.* **19**(90), 297-301 (1965).
- Gentleman, W.M. and Sande, G. "Fast Fourier Transforms--For Fun and Profit." *Proc. AFIPS Fall Joint Computer Conference* **29**, 563-578 (1966).
- Bluestein, L.I. "A Linear Filtering Approach to the Computation of the Discrete Fourier Transform." *IEEE Trans. Audio Electroacoust.* **18**(4), 451-455 (1970).
- Van Loan, C.F. *Computational Frameworks for the Fast Fourier Transform.* SIAM, 1992.
- Winograd, S. "On Computing the Discrete Fourier Transform." *Math. Comp.* **32**, 175-199 (1978).
- Frigo, M. and Johnson, S.G. "The Design and Implementation of FFTW3." *Proc. IEEE* **93**(2), 216-231 (2005).
- Oppenheim, A.V. and Schafer, R.W. *Digital Signal Processing.* Prentice-Hall, 1975.
