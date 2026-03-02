# mdarray

A pure-Python N-dimensional array library with mixed-radix FFT.

Zero runtime dependencies. Strided memory layout, broadcasting, mixed-radix FFT (Temperton staged engine + Bluestein for primes), and basic linear algebra, written entirely in Python. The library began in January 2019 as a ground-up exploration of how N-dimensional array computation actually works—stride arithmetic, broadcast iteration, the matrix factorization of the DFT—and has been periodically retooled since.

The project descends from `fftplusplus` (May 2018), a C++ port of Swarztrauber's FFTPACK with CPython bindings and several pure-Python FFT iterations. That work produced six evolutionary prototypes that eventually became mdarray's core. The FFT module carries forward the same mixed-radix decomposition, now generalized to N dimensions via stride-based fiber extraction.

## Installation

```bash
uv sync
```

Development dependencies (pytest, ruff, mypy, benchmarks):

```bash
uv sync --group dev --group test
```

## Usage

```python
from mdarray import mdarray, zeros, ones, irange, cfft

# Creation
a = irange([3, 4])       # 3x4 array, values 0..11
b = zeros([3, 4])
c = ones([2, 3])

# Arithmetic broadcasts
x = irange(4)             # [0, 1, 2, 3]
y = x + 10               # [10, 11, 12, 13]
z = x * x                # [0, 1, 4, 9]

# FFT
spectrum = cfft(irange(8))

# Linear algebra
from mdarray.linalg import dot, inverse, determinant, solve, qr

A = mdarray(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])
det = determinant(A)
Q, R = qr(A)
```

## Structure

```
src/mdarray/
    array.py             # `mdarray`, creation, manipulation, indexing,
                         # formatting, broadcasting
    core/
        helper.py        # Stride computation, flatten, swap
        reduction.py     # Fold/scan, inner product
        types.py         # inf, nan
        exceptions.py    # IncompatibleDimensions
        logic.py         # Sort, argmax, argmin, where
        math.py          # Element-wise trig and transcendentals
        padding.py       # Array padding
    fft/
        python/
            fft.py       # Temperton staged FFT, Bluestein, N-D gliding
            butterflies.py  # Hand-optimized radix 2-7 butterflies
            twiddle.py   # Twiddle factor precomputation
            factorize.py # Prime factorization
        codegen/
            genfft.py    # Symbolic codelet generator
    linalg/
        math.py          # dot, gaussian_elim, LU, QR, determinant,
                         # inverse, solve, trace, norm
        matrix.py        # diagonal, identity
```

`array.py` is the single source of truth for `mdarray` and every function that creates or manipulates arrays. The import graph is one-directional: `array.py` depends on `core.helper`, `core.exceptions`, `core.reduction`, and `core.types`—all of which are pure utilities with no array dependency. Everything else imports from `array.py`.

## Concepts

### Memory layout — the strided hypercube

`mdarray` stores data in a flat Python list with shape and stride metadata — the **strided hypercube**. Strides are cumulative products of the shape, row-major. The flat index for position `(i_0, i_1, ..., i_{n-1})` is `sum(i_k * strides[k])`. Reshape and transpose modify only the metadata. The same stride system powers iteration, broadcasting, and FFT fiber extraction. See [docs/concepts/memory_layout.md](docs/concepts/memory_layout.md).

### Broadcasting

NumPy-compatible rules: trailing dimensions must be equal or 1. Broadcasting is iterator-based—the `advance()` odometer replays data positions via a repeat counter, so no temporary expanded copies are allocated.

### FFT

The N-dimensional FFT uses **hypercube dimensional gliding**: the flat buffer is treated as an N-dimensional hypercube, and 1-D FFTs are applied along each axis by extracting fibers via stride arithmetic--no reshaping, no transposing, no intermediate buffers. The stride system that powers `advance()` is the same system that enables fiber extraction.

The 1-D engine is Temperton's (1983) staged mixed-radix framework with hand-optimized radix-2, 3, 4, 5, and 7 butterflies. Composite lengths decompose via `factorize(N)` into small prime stages. Prime lengths use Bluestein's chirp-z algorithm for O(N log N) performance. A codelet generator (`fft/codegen/genfft.py`) produces optimized butterflies for arbitrary radices. Provides `cfft`, `ifft`, `fftn`, `ifftn`, and `rfft`. See [docs/concepts/fft.md](docs/concepts/fft.md).

### Linear algebra

Gaussian elimination with partial pivoting, LU and QR decomposition, matrix inverse, determinant, linear solve, trace, and L-p norms.

## Testing

```bash
uv run pytest
uv run pytest --cov
```

107 tests covering core array operations, broadcasting, reductions, FFT correctness (Parseval's theorem, linearity, roundtrip, analytical transforms, Bluestein for large primes, N-D agreement with NumPy), and linear algebra.

## Timeline

- **May 2018** — `fftplusplus`: C++ FFTPACK port with CPython bindings, pure-Python FFT iterations, array prototypes (`array_stuff1–6.py`).
- **Jan 2019** — mdarray initial commit. Daily development through early March: `mdarray` class, stride-based iteration, broadcasting, N-D FFT, pretty printing, linear algebra scaffolding.
- **Feb 2019** — `new_repeats` branch: iterator redesign replacing slice-based repeat with counter-based `advance()` odometer. Concurrent with `fourier-animate` (Fourier series epicycle visualization, separate project).
- **May 2020** — Retooling: cleanup passes, removal of deprecated iterator variants.
- **Sep 2022** — Cleanup: removal of old branches, minor fixes.
- **Mar 2026** — Full modernization: UV migration, Python 3.12+, module consolidation (eliminating all circular dependencies), comprehensive test suite, codelet generator. Restore Temperton staged FFT engine with precomputed twiddle table and DIT butterflies, add Bluestein's chirp-z algorithm for prime lengths, document hypercube dimensional gliding.

## References

- Temperton, C. "Self-Sorting Mixed-Radix Fast Fourier Transforms." *J. Comput. Phys.* **52**, 1–23 (1983).
- Cooley, J.W. and Tukey, J.W. "An Algorithm for the Machine Calculation of Complex Fourier Series." *Math. Comp.* **19**(90), 297–301 (1965).
- Bluestein, L.I. "A Linear Filtering Approach to the Computation of the Discrete Fourier Transform." *IEEE Trans. Audio Electroacoust.* **18**(4), 451–455 (1970).
- Van Loan, C.F. *Computational Frameworks for the Fast Fourier Transform.* SIAM, 1992.
- Frigo, M. and Johnson, S.G. "The Design and Implementation of FFTW3." *Proc. IEEE* **93**(2), 216–231 (2005).
- Golub, G.H. and Van Loan, C.F. *Matrix Computations.* 4th ed., Johns Hopkins, 2013.
- Trefethen, L.N. and Bau, D. *Numerical Linear Algebra.* SIAM, 1997.

## License

MIT
