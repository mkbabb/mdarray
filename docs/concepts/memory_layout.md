# Memory Layout

## Strided arrays

`mdarray` stores all elements in a flat Python list. Shape and stride metadata map N-dimensional indices to flat offsets—the standard apparatus for dense array computation since Iverson's APL (1962), later refined by the NumPy project (van der Walt et al., 2011).

For a shape `[s_0, s_1, ..., s_{n-1}]`, strides are cumulative products:

```
strides[0] = 1
strides[i] = strides[i-1] * shape[i-1]
```

The flat index for position `(i_0, i_1, ..., i_{n-1})`:

```
index = sum(i_k * strides[k] for k in range(ndim))
```

This is row-major (C-order) layout. The first axis varies fastest in memory.

## Why strides matter

Shape and strides are metadata, separate from the data buffer. Operations that only rearrange the *interpretation* of data—without moving the data itself—reduce to metadata edits:

- **Reshape** recomputes strides from a new shape. The flat buffer is untouched if the total element count is preserved.
- **Transpose** swaps the shape and stride entries of two axes.
- **Roll axis** rotates the shape and stride arrays.

In all three cases the underlying list stays put. The operation cost is O(ndim), not O(size).

mdarray doesn't currently implement view semantics (shared data with different strides), so these operations mutate the original array rather than returning a lightweight alias. The stride arithmetic is the same either way—views are a memory-management concern, not a computational one.

## Example

```python
from mdarray import irange

arr = irange([3, 2])  # shape [3, 2], data [0, 1, 2, 3, 4, 5]
# strides = [1, 3]
# arr[col=1, row=0] -> data[1*1 + 0*3] = data[1] = 1
# arr[col=2, row=1] -> data[2*1 + 1*3] = data[5] = 5
```

The stride for axis 0 is 1 (consecutive elements), and for axis 1 it's 3 (skip an entire column's worth of elements). Transposing this array swaps `shape` to `[2, 3]` and `strides` to `[3, 1]`—the same data, read in a different order.

## Implementation

Stride computation lives in `core/helper.py:get_strides()`. The function is small:

```python
def get_strides(shape):
    strides = [1]
    for i in range(1, len(shape)):
        strides.append(strides[i - 1] * shape[i - 1])
    return strides
```

`mdarray.__init__` calls this unless strides are provided explicitly.

## References

- Iverson, K.E. *A Programming Language.* Wiley, 1962.
- van der Walt, S., Colbert, S.C., and Varoquaux, G. "The NumPy Array: A Structure for Efficient Numerical Computation." *Comput. Sci. Eng.* **13**(2), 22–30 (2011).
