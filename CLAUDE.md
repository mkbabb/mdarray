# CLAUDE.md

## What this is

Pure-Python N-dimensional array library. Zero runtime dependencies. Implements strided memory layout, broadcasting, mixed-radix FFT, and linear algebra from scratch.

## Structure

```
src/mdarray/
    array.py             # `mdarray` + creation + manipulation + indexing +
                         # formatting + broadcasting
    core/
        helper.py        # Strides, flatten, swap (no array dependency)
        reduction.py     # Fold/scan, inner product (no array dependency)
        types.py         # inf, nan (no array dependency)
        exceptions.py    # IncompatibleDimensions (no array dependency)
        logic.py         # Sort, argmax, where (imports array)
        math.py          # Trig functions (imports array)
        padding.py       # Pad functions (imports array)
    fft/
        python/
            fft.py       # Temperton staged FFT, Bluestein, N-D gliding
            butterflies.py  # Hand-optimized radix 2-7 butterflies
            twiddle.py   # Twiddle factor precomputation
            factorize.py # Prime factorization
        codegen/
            genfft.py    # Symbolic codelet generator
    linalg/
        math.py          # Gaussian elim, LU, QR, det, inverse, solve
        matrix.py        # diagonal, identity
tests/                   # pytest, 107 tests
```

## Import architecture

One-directional. All circular dependencies eliminated structurally—no `TYPE_CHECKING`, no deferred imports, no `_get_multiarray_class()`, no `try/except ImportError` fallbacks, no `hasattr`/`getattr`/`setattr` hacks.

```
array.py  ->  core.helper, core.exceptions, core.reduction, core.types
core.logic, core.math, core.padding  ->  array.py
fft.python.fft  ->  array.py, core.helper
linalg.math, linalg.matrix  ->  array.py
```

`array.py` consolidates `mdarray` and every function that creates, reshapes, broadcasts, slices, formats, or concatenates arrays. Modules that need `mdarray` import it directly from `array.py`.

## Build and test

```bash
uv sync --group dev --group test
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Conventions

- Python 3.12+. `X | None` not `Optional[X]`, `list[int]` not `List[int]`.
- `__slots__` on `mdarray`.
- Direct top-level imports everywhere. One concrete codepath per operation.
- Zero runtime dependencies. NumPy and SciPy are test-only (oracle implementations).
- Row-major (C-order) layout. Strides are cumulative products of the shape.

## `mdarray` iteration

The `advance()` method is a mixed-radix counter with carry propagation. Three parallel arrays track state:

- `_axis_counter[i]` — current offset along axis i (in stride units).
- `_rept_counter[i]` — broadcast replay counter for axis i.
- `_was_advanced[i]` — carry flag, used by formatting and concatenation to detect row/page boundaries.

`_effective_size` = product of `shape[i] * (repeats[i] + 1)`. A repeat of 0 means no extra copies.

## FFT

Temperton's (1983) staged mixed-radix framework operating on flat Python lists. Decomposes N into prime factors, applies DIT butterflies at increasing stride with precomputed twiddle table, input digit-reversed for natural-order output. Hand-optimized radix 2, 3, 4, 5, 7 butterflies; generic `radixg` for other prime factors within composites. Prime-length N uses Bluestein's chirp-z algorithm (O(N log N) via power-of-2 convolution). N-D transforms use hypercube dimensional gliding: extract fibers along each axis via stride arithmetic, apply 1-D FFT, write back. No reshaping or transposing.

## Broadcasting

`generate_broadcast_shape()` computes the output shape and per-array repeat counts. `broadcast_nary()` zips iterators with repeat-aware advance and applies a function element-wise. Repeat values are `target_size - 1` (number of extra copies, not total iterations).

## Known limitations

- No dtype system—data is Python lists of arbitrary objects.
- No view semantics—reshape and transpose mutate in-place.
- Slice support is basic (integer indices, limited slice objects).

## Branch

Active branch: `master`.
