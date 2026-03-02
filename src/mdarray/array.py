from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Iterator
from functools import reduce
from typing import Any

from .core import exceptions, helper
from .core.helper import get_strides, pair_wise, swap
from .core.reduction import inner_product
from .core.types import inf

# ---------------------------------------------------------------------------
# mdarray
# ---------------------------------------------------------------------------


class mdarray:
    __slots__ = (
        "_axis_counter",
        "_data",
        "_index",
        "_mdim",
        "_order",
        "_pos",
        "_repeats",
        "_rept_counter",
        "_shape",
        "_size",
        "_stride_shape",
        "_strides",
        "_was_advanced",
    )

    def __init__(
        self,
        data: list[Any] | mdarray | None = None,
        shape: list[int] | None = None,
        strides: list[int] | None = None,
        mdim: int | None = None,
        size: int | None = None,
        order: str | None = None,
    ) -> None:
        if not shape and not size and not data:
            raise TypeError("Must provide at least one of: data, shape, or size")
        elif shape and not size:
            self._shape = list(shape)
            self._size = reduce(lambda x, y: x * y, shape)
        elif size and not shape:
            self._size = size
            self._shape = [size]
        else:
            self._size = size if size else reduce(lambda x, y: x * y, shape)  # type: ignore
            self._shape = list(shape) if shape else [self._size]

        self._mdim = len(self._shape)

        if strides:
            self._strides = list(strides)
        else:
            self._strides = get_strides(self._shape)

        self._order = order if order else "C"

        if data is None:
            self._data: list[Any] = []
        elif isinstance(data, mdarray):
            self._data = list(data._data)
        else:
            self._data = list(data)

        self._stride_shape = pair_wise(self._shape, self._strides, operator.mul)

        self._axis_counter = [0] * self._mdim
        self._was_advanced = [False] * self._mdim

        self._rept_counter = [0] * self._mdim
        self._repeats = [0] * self._mdim

        self._pos = 0
        self._index = 0

    @property
    def shape(self) -> list[int]:
        return self._shape

    @property
    def strides(self) -> list[int]:
        return self._strides

    @property
    def mdim(self) -> int:
        return self._mdim

    @property
    def size(self) -> int:
        return self._size

    @property
    def data(self) -> list[Any]:
        return self._data

    @data.setter
    def data(self, value: list[Any]) -> None:
        self._data = value

    @property
    def axis_counter(self) -> list[int]:
        return self._axis_counter

    @property
    def was_advanced(self) -> list[bool]:
        return self._was_advanced

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def index(self) -> int:
        return self._index

    @property
    def repeats(self) -> list[int]:
        return self._repeats

    @repeats.setter
    def repeats(self, value: list[int]) -> None:
        self._repeats = value
        self._rept_counter = [0] * self._mdim

    # --- Iteration ---

    def advance(self, step: int = 1) -> int:
        i = 0
        while i < step:
            if self._rept_counter[0] < self._repeats[0]:
                self._rept_counter[0] += 1
            else:
                self._rept_counter[0] = 0
                self._axis_counter[0] += self.strides[0]

            for j in range(1, self.mdim):
                if self._axis_counter[j - 1] >= self._stride_shape[j - 1]:
                    if self._rept_counter[j] == self._repeats[j]:
                        self._rept_counter[j] = 0
                        self._axis_counter[j - 1] = 0
                        self._axis_counter[j] += self.strides[j]
                        self._was_advanced[j] = True
                    else:
                        self._rept_counter[j] += 1
                        self._axis_counter[j - 1] = 0
                        self._was_advanced[j] = True
                else:
                    self._was_advanced[j] = False
            i += 1

        self._index = sum(self._axis_counter)
        self._pos += step

        return self._index

    def at(self, pos: list[int] | int) -> mdarray:
        if isinstance(pos, list) and len(pos) == self.mdim:
            self._pos = unravel(pos, self.shape, self.strides)
        else:
            self._pos = pos  # type: ignore

        if self._pos == 0:
            for i in range(self._mdim):
                self._axis_counter[i] = 0
                self._was_advanced[i] = False
            self._index = 0
            self._pos = 0
        else:
            ravel(self._pos, self._shape, self._strides, self._axis_counter)

            for i in range(1, self._mdim):
                self.was_advanced_before(i)

            self._index = sum(self._axis_counter)

        return self

    def __next__(self) -> mdarray:
        if self._index < self._size:
            self.advance(1)
            return self
        else:
            raise StopIteration

    @property
    def _effective_size(self) -> int:
        sizes = [self._shape[i] * (self._repeats[i] + 1) for i in range(self._mdim)]
        return reduce(operator.mul, sizes, 1)

    def __iter__(self) -> Iterator[Any]:
        self.at(0)
        eff_size = self._effective_size
        for _ in range(eff_size):
            yield self
            try:
                self.__next__()
            except StopIteration:
                break

    def zero_axes_before(self, axis: int) -> bool:
        return all(not (i != axis and self._rept_counter[i] != 0) for i in range(axis))

    def was_advanced_before(self, axis: int) -> bool:
        for i in range(axis):
            if self._axis_counter[i] != (self._stride_shape[i] - 1):
                self._was_advanced[axis] = False
                return False
        self._was_advanced[axis] = True
        return True

    # --- Shape manipulation ---

    def reshape(self, new_shape: list[int]) -> mdarray:
        reshape(self, new_shape)
        return self

    def T(self, axis1: int = 0, axis2: int = 1) -> mdarray:
        transpose(self, axis1, axis2)
        return self

    def flatten(self, order: int = -1) -> mdarray:
        flatten(self, order)
        return self

    def to_list(self) -> list[Any]:
        return make_nested_list(self)

    # --- Operator overloads ---

    def _apply_binary(self, other: Any, op: Callable) -> mdarray:
        other = tomdarray(other)
        return broadcast_nary(self, other, func=lambda args: op(args[0], args[1]))

    def _apply_unary(self, op: Callable) -> mdarray:
        arr_out = zeros(shape=list(self._shape))
        for i in range(self._size):
            arr_out._data[i] = op(self._data[i])
        return arr_out

    def __add__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.add)

    def __radd__(self, other: Any) -> mdarray:
        return self._apply_binary(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.sub)

    def __rsub__(self, other: Any) -> mdarray:
        return self._apply_binary(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.mul)

    def __rmul__(self, other: Any) -> mdarray:
        return self._apply_binary(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> mdarray:
        return self._apply_binary(other, lambda a, b: b / a)

    def __floordiv__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.floordiv)

    def __mod__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.mod)

    def __pow__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.pow)

    def __rpow__(self, other: Any) -> mdarray:
        return self._apply_binary(other, lambda a, b: b**a)

    def __neg__(self) -> mdarray:
        return self._apply_unary(operator.neg)

    def __abs__(self) -> mdarray:
        return self._apply_unary(abs)

    # --- Comparison operators ---

    def __eq__(self, other: Any) -> mdarray:  # type: ignore[override]
        return self._apply_binary(other, operator.eq)

    def __ne__(self, other: Any) -> mdarray:  # type: ignore[override]
        return self._apply_binary(other, operator.ne)

    def __lt__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.lt)

    def __le__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.le)

    def __gt__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.gt)

    def __ge__(self, other: Any) -> mdarray:
        return self._apply_binary(other, operator.ge)

    # --- Indexing ---

    def __getitem__(self, key: Any) -> mdarray:
        key = list(key) if isinstance(key, tuple) else [key]
        return slice_array(key, self, None, False)

    def __setitem__(self, key: Any, value: Any) -> None:
        key = list(key) if isinstance(key, tuple) else [key]
        slice_array(key, self, value, True)

    # --- Representation ---

    def __str__(self) -> str:
        return print_array(self)

    def __repr__(self) -> str:
        return f"mdarray({print_array(self)}, shape={self._shape})"

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def tomdarray(arr: Any) -> mdarray:
    if isinstance(arr, mdarray):
        return arr
    if isinstance(arr, (list, tuple)):
        arr, _mdim, shape = helper.flatten_list(arr, order=-1)
        return mdarray(shape=shape, data=arr)
    if isinstance(arr, (int, float, complex, str)):
        return mdarray(size=1, data=[arr])
    arr = list(arr)
    return mdarray(size=len(arr), data=arr)


def zeros(
    shape: list[int] | None = None,
    size: int | None = None,
    dtype: Any = None,
    order: str | None = None,
) -> mdarray:
    arr_out = mdarray(shape=shape, size=size, order=order)
    arr_out._data = [0] * arr_out.size
    return arr_out


def ones(
    shape: list[int] | None = None,
    size: int | None = None,
    dtype: Any = None,
    order: str | None = None,
) -> mdarray:
    arr_out = mdarray(shape=shape, size=size, order=order)
    arr_out._data = [1] * arr_out.size
    return arr_out


def full(
    shape: list[int] | None = None,
    fill: Any = 0,
    fill_value: Any = None,
    size: int | None = None,
    dtype: Any = None,
    order: str | None = None,
) -> mdarray:
    val = fill_value if fill_value is not None else fill
    arr_out = mdarray(shape=shape, size=size, order=order)
    arr_out._data = [val] * arr_out.size
    return arr_out


def irange(size: int | list[int]) -> mdarray:
    if isinstance(size, list):
        shape = size
        size = reduce(lambda x, y: x * y, size)
    else:
        shape = [size]
    data = list(range(size))
    return mdarray(shape=shape, data=data)


def linear_range(
    start: int | float,
    stop: int | float,
    size: int | None = None,
) -> mdarray:
    if not size:
        size = int(stop - start)
    arr_out = zeros(shape=[size])

    if isinstance(start, int) and isinstance(stop, int):
        step = (stop - start) // size
    else:
        step = (stop - start) / size

    i = start
    j = 0
    while j < size:
        arr_out.data[j] = i
        j += 1
        i += step
    return arr_out


def log_range(
    start: int | float,
    stop: int | float,
    base: int | float,
    size: int | None = None,
) -> mdarray:
    return base ** linear_range(start, stop, size)


# ---------------------------------------------------------------------------
# Tiling and grid
# ---------------------------------------------------------------------------


def _sort_axes(raxes: list[int], repts: list[int], mdim: int) -> None:
    ndim = len(raxes)
    if mdim > ndim:
        pad = [1] * (mdim - ndim)
        raxes += pad
        repts += pad

    def recurse(ix: int) -> None:
        for i in range(ix, mdim):
            raxis = raxes[i]
            rept = repts[i]

            if raxis != i and rept != 1:
                swap(raxes, i, raxis)
                swap(repts, i, raxis)
                if raxis > i:
                    recurse(i + 1)

    recurse(0)


def repeat(
    arr: mdarray,
    raxes: list[int],
    repts: list[int],
) -> mdarray:
    mdim = arr.mdim
    _sort_axes(raxes, repts, mdim)
    new_shape = list(arr.shape)

    for i in range(mdim):
        rept = repts[i]
        raxis = raxes[i]
        new_shape[raxis] *= rept

    arr.repeats = repts
    arr_out = zeros(shape=new_shape)

    for n, _ in enumerate(arr):
        arr_out.data[n] = arr.data[arr.index]

    return arr_out


def meshgrid_internal(
    arrs: list | tuple,
    _iter: bool = True,
) -> list[mdarray]:
    sizes = list(map(len, arrs))
    mdim = len(arrs)

    arrs_out: list[mdarray] = [None] * mdim  # type: ignore
    for i in range(mdim):
        slc = [1] * mdim
        slc[i] = sizes[i]
        if _iter:
            arrs_out[i] = mdarray(shape=slc)
        else:
            arrs_out[i] = tomdarray(arrs[i]).reshape(slc)

    return arrs_out


def ix_meshgrid(*arrs: Any) -> list[mdarray]:
    return meshgrid_internal(arrs, True)


def dense_meshgrid(*arrs: Any) -> list[mdarray]:
    return meshgrid_internal(arrs, False)


def meshgrid(*arrs: Any) -> list[mdarray]:
    arrs_mesh = meshgrid_internal(arrs, False)
    return broadcast(*arrs_mesh)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


def generate_broadcast_shape(
    *arrs: mdarray,
) -> tuple[list[int], list[list[int]]]:
    arrs_t = tuple(arrs)
    ndim = len(arrs_t)

    shapes = [list(i.shape) for i in arrs_t]
    mdims = [i.mdim for i in arrs_t]
    mdim = max(mdims)

    for i in range(ndim):
        if mdims[i] < mdim:
            shapes[i] += [1] * (mdim - mdims[i])
        arrs_t[i].reshape(shapes[i])

    repts = [[0] * mdim for _ in range(ndim)]
    new_shape = list(shapes[0])

    for i in range(mdim):
        axis_i = shapes[0][i]
        j = 1

        while j < ndim:
            axis_j = shapes[j][i]
            if axis_i == 1 and axis_j > 1:
                axis_i = axis_j
                repts[0][i] = axis_i - 1
                j = 0
            elif axis_i > 1 and axis_j == 1:
                repts[j][i] = axis_i - 1
            elif axis_i != axis_j:
                raise exceptions.IncompatibleDimensions(
                    f"Cannot broadcast shapes: axis {i} has sizes {axis_i} and {axis_j}"
                )
            j += 1

        new_shape[i] = axis_i

    return new_shape, repts


def broadcast_iter(*arrs: mdarray) -> list[int]:
    new_shape, repts = generate_broadcast_shape(*arrs)
    for i in range(len(arrs)):
        arrs[i].repeats = repts[i]
    return new_shape


def broadcast_nary(
    *arrs: mdarray,
    func: Callable[[list[Any]], Any],
) -> mdarray:
    new_shape = broadcast_iter(*arrs)
    arr_out = zeros(new_shape)
    ndim = len(arrs)
    fargs: list[Any] = [0] * ndim

    for n, _items in enumerate(zip(*arrs, strict=False)):
        for m in range(ndim):
            fargs[m] = arrs[m].data[arrs[m].index]
        arr_out.data[n] = func(fargs)
    return arr_out


def broadcast_toshape(arr: mdarray, shape: list[int]) -> mdarray:
    arr_shape = mdarray(shape=shape)
    new_shape = broadcast_iter(arr, arr_shape)
    arr_out = zeros(new_shape)

    for n, _ in enumerate(arr):
        arr_out.data[n] = arr.data[arr.index]
    return arr_out


def broadcast(*arrs: mdarray) -> list[mdarray]:
    ndim = len(arrs)
    new_shape = broadcast_iter(*arrs)
    arrs_out = [zeros(new_shape) for _ in range(ndim)]

    for n, _items in enumerate(zip(*arrs, strict=False)):
        for m in range(ndim):
            arrs_out[m].data[n] = arrs[m].data[arrs[m].index]
    return arrs_out


# ---------------------------------------------------------------------------
# Manipulation
# ---------------------------------------------------------------------------


def reshape(arr: mdarray, new_shape: list[int]) -> None:
    new_shape = list(new_shape)
    mdim = len(new_shape)
    new_size = reduce(lambda x, y: x * y, new_shape)

    if new_size != arr.size:
        raise exceptions.IncompatibleDimensions(
            "The desired shape is incompatible with the current array's shape."
        )
    else:
        arr._shape = new_shape
        arr._mdim = mdim
        arr._strides = get_strides(new_shape)

    arr._stride_shape = pair_wise(arr._shape, arr._strides, operator.mul)

    arr._axis_counter = [0] * arr._mdim
    arr._was_advanced = [False] * arr._mdim

    arr._rept_counter = [0] * arr._mdim
    arr._repeats = [0] * arr._mdim

    arr._pos = 0
    arr._index = 0


def make_mdim(arr: mdarray, ndim: int) -> None:
    if arr.mdim < ndim:
        new_shape = arr.shape + [1] * (ndim - arr.mdim)
        reshape(arr, new_shape)


def transpose(arr: mdarray, axis1: int = 0, axis2: int = 1) -> None:
    mdim = arr.mdim

    if axis1 < 0:
        axis1 += mdim
    if axis2 < 0:
        axis2 += mdim

    maxis = max(axis1, axis2)
    if maxis > mdim - 1:
        paxis = maxis - (mdim - 1)
        reshape(arr, arr.shape + [1] * paxis)

    swap(arr.strides, axis1, axis2)
    swap(arr.shape, axis1, axis2)


def swap_axis(arr: mdarray, axis1: int = 0, axis2: int = 1) -> None:
    transpose(arr, axis1, axis2)


def roll_axis(arr: mdarray, axis: int, iterations: int = 1) -> None:
    mdim = arr.mdim
    if axis < 0:
        axis += mdim
    helper.roll_array(arr.shape, axis, iterations)
    helper.roll_array(arr.strides, axis, iterations)


def flatten(arr: mdarray, order: int = -1) -> None:
    mdim = arr.mdim

    if order < 0:
        order += mdim
    elif order == 0:
        return

    new_mdim = arr.mdim - order
    new_shape = [0] * (arr.mdim - order)

    for i in range(new_mdim):
        new_shape[i] = arr.shape[i]

    red = 1
    for i in range(new_mdim - 1, mdim):
        red *= arr.shape[i]

    new_shape[-1] = red
    reshape(arr, new_shape)


def make_nested_list(arr: mdarray) -> list[Any]:
    tmp: list[Any] = []
    nests: list[list] = [[] for _ in range(arr.mdim - 1)]

    arr.at(0)
    for i in range(arr.size):
        tmp.append(arr.data[arr.index])

        if i < arr.size - 1:
            try:
                next(arr)
            except StopIteration:
                break

        for j in range(1, arr.mdim):
            if arr.was_advanced[j]:
                if j == 1:
                    nests[0].append(tmp)
                    tmp = []
                else:
                    nests[j - 1].append(nests[j - 2])
                    nests[j - 2] = []

    if tmp and nests:
        nests[0].append(tmp)

    arr.at(0)
    return nests[-1] if nests else tmp


def _confirm_concat_shape(
    arrs: tuple[mdarray, ...],
    caxis: int,
    mdim: int,
    ndim: int,
    new_shape: list[int],
) -> list[int]:
    new_shape[caxis] = 0

    for i in range(ndim):
        arr_i = arrs[i]
        if mdim != arr_i.mdim:
            raise ValueError(f"The dimensions of array 1 != the dimensions of array {i}!")

        for j in range(mdim):
            if j != caxis and new_shape[j] != arr_i.shape[j]:
                raise ValueError("All axes but caxis must be equivalent to concatenate the arrays.")

        new_shape[caxis] += arr_i.shape[caxis]

    return new_shape


def concatenate(*arrs: mdarray, caxis: int = -1) -> mdarray:
    arrs_t = tuple(arrs)
    new_shape = _confirm_concat_shape(
        arrs_t, caxis, arrs_t[0].mdim, len(arrs_t), list(arrs_t[0].shape)
    )

    arr_out = zeros(shape=new_shape)

    caxis_resolved = -1 if caxis >= arr_out.mdim - 1 else caxis

    j = 0
    for arr in arrs_t:
        arr.at(0)

    while j < arr_out.size:
        for arr in arrs_t:
            while not arr.was_advanced[caxis_resolved + 1] and arr.index < arr.size:
                arr_out.data[j] = arr.data[arr.index]
                j += 1
                try:
                    next(arr)
                except StopIteration:
                    break
            arr._was_advanced[caxis_resolved + 1] = False

    for arr in arrs_t:
        arr.at(0)

    return arr_out


def hstack(*arrs: mdarray) -> mdarray:
    return concatenate(*arrs, caxis=0)


def vstack(*arrs: mdarray) -> mdarray:
    return concatenate(*arrs, caxis=1)


def dstack(*arrs: mdarray) -> mdarray:
    return concatenate(*arrs, caxis=2)


def tile(arr: mdarray, tiles: list[int]) -> mdarray:
    mdim = arr.mdim
    ndim = len(tiles)
    if ndim > mdim:
        new_shape = arr.shape + [1] * (ndim - mdim)
        arr = arr.reshape(new_shape)

    arr_i = arr
    for i in range(ndim):
        tile_i = tiles[i]
        tile_arrs = [arr_i] * tile_i
        arr_i = concatenate(*tile_arrs, caxis=i)
    return arr_i


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def ravel(
    ix: int,
    shape: list[int],
    strides: list[int] | None = None,
    mdim_ixs: list[int] | None = None,
) -> list[int]:
    mdim = len(shape)

    if not strides:
        strides = get_strides(shape)
    if not mdim_ixs:
        mdim_ixs = [0] * mdim

    for i in range(mdim):
        stride = strides[mdim - (i + 1)]
        j = ix // stride
        ix -= j * stride
        mdim_ixs[mdim - (i + 1)] = j * stride

    return mdim_ixs


def unravel(
    mdim_ix: list[int],
    shape: list[int],
    strides: list[int] | None = None,
) -> int:
    if not strides:
        strides = get_strides(shape)
    return inner_product(strides, mdim_ix)


def unravel_dense(
    dense_ixs: list[mdarray],
    arr_in: mdarray,
    arr_out: mdarray | None = None,
    setter: bool = False,
) -> None:
    strides = arr_in.strides
    for n, i in enumerate(zip(*dense_ixs, strict=False)):
        ix_i = 0
        for m, _j in enumerate(i):
            ix_i += dense_ixs[m].data[dense_ixs[m].index] * strides[m]
        if setter:
            arr_in.data[ix_i] = arr_out.data[n]  # type: ignore
        else:
            arr_out.data[n] = arr_in.data[ix_i]  # type: ignore


def expand_indicies(
    slc: Any,
    arr: mdarray,
) -> tuple[list, list[int], bool]:
    try:
        slc = list(slc)
    except TypeError:
        slc = [slc]
    ndim = len(slc)
    oned = True
    new_shape = [0] * ndim

    for i in range(ndim):
        arr_i = slc[i]
        if not isinstance(arr_i, mdarray):
            arr_i = irange(arr.shape[i]) if arr_i == inf or arr_i is Ellipsis else tomdarray(slc[i])

        new_shape[i] = arr_i.size
        oned = False if arr_i.mdim > 1 else oned
        slc[i] = arr_i

    return slc, new_shape, oned


def slice_array(
    slc: Any,
    arr_in: mdarray,
    arr_out: mdarray | None,
    setter: bool = True,
) -> mdarray:
    slc, new_shape, oned = expand_indicies(slc, arr_in)

    if oned:
        slc = ix_meshgrid(*slc)
        new_shape = broadcast_iter(*slc)
    else:
        new_shape = broadcast_iter(*slc)

    arr_out = zeros(new_shape) if not arr_out else tomdarray(arr_out)

    if arr_in.shape != arr_out.shape:
        arr_out = broadcast_toshape(arr_out, new_shape)

    unravel_dense(slc, arr_in, arr_out, setter)

    return arr_out


def indicies(arr: mdarray, ixs: Any, axis: int = -1) -> mdarray:
    mdim = arr.mdim
    shape = arr.shape

    ranges: list[Any] = [0] * mdim
    for i in range(mdim):
        ranges[i] = list(range(shape[i]))

    ix_grid = dense_meshgrid(*ranges)
    ix_grid[axis] = ixs

    return arr[tuple(ix_grid)]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

MAX_CHARS_LINE = 50
SAVED_CHAR = 3


def trim_string(s: str, sep: str) -> str:
    sep_len = len(sep)

    if len(s) >= MAX_CHARS_LINE:
        s_re = list(re.finditer(sep, s))
        N = len(s_re)

        if N >= (2 * SAVED_CHAR + 1):
            start = s_re[SAVED_CHAR - 1].start()
            stop = s_re[N - (SAVED_CHAR)].start()

            s = f"{s[:start]} ... {s[stop + sep_len :]}"

    return s


def print_array(
    arr: mdarray,
    sep: str | None = ", ",
    formatter: Callable[[list, int], str] | None = None,
) -> str:
    if not formatter:

        def formatter(x: list, y: int) -> str:
            return f"{x[y]}"

    if not sep:
        sep = ", "

    t_ix = arr.index

    s = "[" * (arr.mdim)

    arr.at(0)
    for i in range(arr.size):
        s += formatter(arr.data, arr.index)

        if i < arr.size - 1:
            try:
                next(arr)
            except StopIteration:
                s += "]" * arr.mdim
                break
            ix = 0
            for j in range(1, arr.mdim):
                if arr.was_advanced[j]:
                    ix += 1

            if ix > 0:
                s += "]" * ix
                s += "\n" * ix
                s += " " * (arr.mdim - ix)
                s += "[" * ix
            else:
                s += sep
        else:
            s += "]" * arr.mdim
    arr.at(t_ix)
    return s


def pad_array_fmt(arr: mdarray) -> Callable[[Any], str]:
    max_len = len(str(max(arr.data, key=lambda x: len(str(x)))))

    def fmter(x: Any) -> str:
        pad_left = math.ceil((max_len - len(str(x))) / 2)
        pad_right = math.floor((max_len - len(str(x))) / 2)
        return f"{' ' * pad_left}{x}{' ' * pad_right}"

    return fmter
