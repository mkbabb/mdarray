# Iteration

## The advance() odometer

`mdarray.__next__()` calls `advance(1)`, a generalized mixed-radix counter with carry propagation. It tracks position across N dimensions using three parallel arrays:

- `_axis_counter[i]` — current offset along axis i, in stride units (0, stride, 2*stride, ...).
- `_rept_counter[i]` — how many times the current position along axis i has been replayed.
- `_was_advanced[i]` — whether axis i just carried.

The innermost axis (index 0) increments on every call. When `_axis_counter[0]` reaches `_stride_shape[0]` (= `shape[0] * strides[0]`), it resets to 0 and carries to axis 1. Before carrying, the repeat counter is checked—if `_rept_counter[1] < _repeats[1]`, the carry replays the same data range instead of advancing.

The flat data index is `sum(_axis_counter)`.

This mechanism was introduced in February 2019 on the `new_repeats` branch, replacing an earlier approach that used slice objects for repetition. The counter-based design turned out to be simpler and more general: broadcasting, `repeat()`, `tile()`, and the `make_nested_list()` structure builder all reduce to setting repeat counters and iterating.

## How broadcasting reduces to iteration

When two arrays are broadcast together:

1. `generate_broadcast_shape()` computes the output shape and assigns `repeats` per axis per array.
2. Both arrays are iterated in lockstep (via `zip`).
3. Array A might have `repeats=[0, 2]`: no repeat on axis 0, two extra replays on axis 1. Its single row of data is yielded 3 times.
4. Array B might have `repeats=[3, 0]`: each column replayed 4 times, no repeat on rows.

The output loop sees both arrays producing `_effective_size` elements, synchronized by the odometer. `_effective_size` is `product(shape[i] * (repeats[i] + 1))`.

## Yielding self

`__iter__` yields `self` (the `mdarray` object), not data values directly. The caller reads `self.data[self.index]` to get the current element—the index reflects the actual data position accounting for strides, while the iteration position accounts for repeats:

```python
for _ in arr:
    value = arr.data[arr.index]
```

This is a deliberate design choice. The iterator advances the odometer state; the caller decides what to do with the current position. The pattern enables operations like `concatenate()` and `broadcast_nary()` to interleave multiple iterators, each maintaining independent state, without materializing intermediate arrays.

## was_advanced

The `_was_advanced` flags are the key to multidimensional output structure. When formatting a 3-D array, axis 1 advancing means "start a new row," axis 2 advancing means "start a new page." `make_nested_list()` uses the same flags to build nested Python lists from flat iteration. `concatenate()` uses them to interleave arrays along the concatenation axis.

The flags are set during carry propagation in `advance()` and cleared on the next call when no carry occurs. They provide a lightweight event system that converts the flat iteration stream into structured multidimensional output.

## Connection to dimensional gliding

The `advance()` odometer and the N-D FFT's fiber extraction are two faces of the same coin. Both navigate the strided hypercube via stride arithmetic:

- **advance()** traverses all elements in a fixed order, incrementing axis counters with carry propagation. It is a sequential scan of the full hypercube.
- **Dimensional gliding** (used by `fftn`) fixes one axis, enumerates all fibers along that axis by iterating over orthogonal coordinates, and steps along the fiber at that axis's stride. It is a structured traversal that decomposes the hypercube into independent 1-D slices.

Both algorithms rely on the same stride formula `offset = sum(i_k * stride_k)` and the same cumulative-product strides from `get_strides()`. The hypercube is the universal data structure; stride arithmetic is the universal traversal primitive.
