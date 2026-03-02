# Broadcasting

## Rules

Two shapes are broadcast-compatible when, comparing dimensions from the trailing end:

1. The dimensions are equal, or
2. One of them is 1.

The output shape takes the maximum along each axis. An array with a size-1 dimension gets repeated along that axis to match the larger dimension.

These are the standard NumPy broadcasting rules, formalized in the array API specification.

## Iterator-based broadcasting

Most array libraries implement broadcasting by allocating expanded temporary copies. mdarray takes a different approach: broadcasting is built into the iterator via repeat counters.

When `broadcast_iter()` prepares arrays for a broadcast operation:

1. `generate_broadcast_shape()` computes the output shape and a per-array `repeats` list.
2. Each array's `_repeats` property is set. A repeat value of `k` means `k` extra copies—so repeat=0 is no repetition, repeat=2 means the axis is traversed 3 times total.
3. The `advance()` odometer checks `_rept_counter` before incrementing the axis counter. When a repeat is active, the counter resets without advancing into the data, replaying the same positions.

Broadcast operations allocate only the output array. Input arrays are traversed with virtual repetition.

## The repeat mechanism

The `mdarray` iterator is a generalized odometer with carry. For each axis:

- `_axis_counter[i]` tracks the current position (in stride units).
- `_rept_counter[i]` tracks how many times the current slice has been replayed.
- `_was_advanced[i]` flags whether a carry propagated to this axis.

When `_rept_counter[i] < _repeats[i]`, the counter increments without advancing `_axis_counter[i]`. The data index stays the same while the logical output position advances. A shape-`[4]` array broadcasting to `[4, 3]` produces 12 values this way—each of the 4 elements yielded 3 times—without copying anything.

This design emerged from the `new_repeats` branch (February 2019), which replaced an earlier slice-based repeat mechanism with the counter-based odometer. The name stuck.

## Example

```python
from mdarray import irange, ones

a = irange(4)            # shape [4], data [0, 1, 2, 3]
b = ones([4, 3])          # shape [4, 3], all ones

c = a + b                # a broadcast to [4, 3]: each element repeated 3 times
# c.data = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
```

Here `a` has `repeats=[0, 2]` (no repeat on axis 0, 2 extra copies on axis 1). Its 4 elements are each yielded 3 times, synchronized with `b`'s natural iteration over 12 elements.

## Connection to repeat and tile

The `repeat()` and `tile()` functions use the same repeat infrastructure. `repeat(arr, n, axis)` sets the repeat counter on the specified axis and materializes the result. `tile(arr, reps)` sets repeats on every axis. Broadcasting is the implicit version of the same operation—repeats are set automatically based on shape compatibility, and the result is computed element-wise without materializing the expanded arrays.
