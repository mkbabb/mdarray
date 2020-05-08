from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.creation import (broadcast_iter, broadcast_toshape, ix_meshgrid,
                           irange, tomdarray, zeros)

from core.exceptions import IncompatibleDimensions
from core.helper import get_strides
from core.reduction import inner_product
from core.types import inf, nan
from MultiArray import MultiArray

__all__ = ["ravel", "unravel",
           "unravel_dense", "slice_array", "expand_indicies",
           "indicies"]


'''
Raveling and unraveling indicies:
'''


def ravel(ix: int,
          shape: int,
          strides: Optional[List[int]] = None,
          mdim_ixs: Optional[List[int]] = None) -> List[int]:
    mdim = len(shape)

    if (not strides):
        strides = get_strides(shape)
    if (not mdim_ixs):
        mdim_ixs = [0] * mdim

    for i in range(mdim):
        stride = strides[mdim - (i + 1)]
        j = 1
        while True:
            stride_j = stride * j
            if stride_j > ix:
                j -= 1
                stride_j = stride * j
                break
            elif stride_j == ix:
                break
            else:
                j += 1
        ix -= stride_j
        mdim_ixs[mdim - (i + 1)] = stride_j

    return mdim_ixs


def unravel(mdim_ix: List[int],
            shape: List[int],
            strides: Optional[List[int]] = None) -> int:
    if (not strides):
        strides = get_strides(shape)
    return inner_product(strides, mdim_ix)


'''
M-d array slicing:
'''


def unravel_dense(dense_ixs: List[MultiArray],
                  arr_in: MultiArray,
                  arr_out: Optional[MultiArray] = None,
                  setter: bool = False) -> MultiArray:
    strides = arr_in.strides
    for n, i in enumerate(zip(*dense_ixs)):
        ix_i = 0
        for m, j in enumerate(i):
            ix_i += j.arr.data[j.index] * strides[m]
        if setter:
            arr_in.data[ix_i] = arr_out.data[n]
        else:
            arr_out.data[n] = arr_in.data[ix_i]


def expand_indicies(slc, arr):
    try:
        slc = list(slc)
    except TypeError:
        slc = [slc]
    ndim = len(slc)
    oned = True
    new_shape = [0] * ndim

    for i in range(ndim):
        arr_i = slc[i]
        if not isinstance(arr_i, MultiArray):
            if arr_i == inf or arr_i == Ellipsis:
                arr_i = irange(arr.shape[i])
            else:
                arr_i = tomdarray(slc[i])

        new_shape[i] = arr_i.size
        oned = False if arr_i.mdim > 1 else oned
        slc[i] = arr_i

    return slc, new_shape, oned


def slice_array(slc, arr_in, arr_out, setter=True):
    slc, new_shape, oned = expand_indicies(slc, arr_in)
    order = arr_in.order

    if oned:
        slc = ix_meshgrid(*slc)
        slc = broadcast_iter(*slc)
    else:
        slc = broadcast_iter(*slc)
        new_shape = slc[0].shape

    if not arr_out:
        arr_out = zeros(new_shape, order=order, dtype=arr_in.dtype)
    else:
        arr_out = tomdarray(arr_out)

    if arr_in.shape != arr_out.shape:
        arr_out = broadcast_toshape(arr_out, new_shape)

    unravel_dense(slc, arr_in, arr_out, setter)

    return arr_out


'''
End M-d array slicing
'''


def indicies(arr, ixs, axis=-1):
    mdim = arr.mdim
    shape = arr.shape

    ranges = [0] * mdim
    for i in range(mdim):
        ranges[i] = list(range(shape[i]))

    ix_grid = dense_meshgrid(*ranges)
    ix_grid[axis] = ixs

    return arr[ix_grid]
