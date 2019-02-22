from functools import reduce

import mdarray as md
from core.creation import (broadcast_arrays, broadcast_toshape, dense_meshgrid,
                           irange, tomdarray, zeros)
from core.exceptions import IncompatibleDimensions
from core.helper import get_strides
from core.types import inf, nan

__all__ = ["ravel_internal", "ravel", "unravel",
           "unravel_dense", "slice_array", "expand_indicies",
           "indicies"]


'''
Raveling and unraveling indicies:
'''


def ravel_internal(ix, mdim_ix_i, mdim, strides):
    for j in range(mdim):
        stride = strides[mdim - (j + 1)]
        k = 1

        while True:
            stride_k = stride * k
            if stride_k > ix:
                k -= 1
                stride_k = stride * k
                break
            elif stride_k == ix:
                break
            else:
                k += 1

        ix -= stride_k
        mdim_ix_i[mdim - (j + 1)] = k

    return mdim_ix_i


def ravel(ixs, shape):
    if isinstance(shape, md.mdarray):
        strides = shape.strides
        size = shape.size
        mdim = shape.mdim
    else:
        strides = get_strides(shape)
        size = reduce(lambda x, y: x * y, shape)
        mdim = len(shape)

    ixs = tuple(ixs)
    ndim = len(ixs)
    mdim_ixs = [[0] * mdim] * ndim

    for i in range(ndim):
        ix = ixs[i]
        mdim_ix_i = [0] * mdim
        if ix > size:
            raise IncompatibleDimensions(
                "The raveled index is too large to unravel using the provided shape!")
        elif ix == 0:
            mdim_ixs[i] = mdim_ix_i
        else:
            ix += size if ix < 0 else 0
            mdim_ixs[i] = ravel_internal(ix, mdim_ix_i, mdim, strides)

    return mdim_ixs


def unravel(mdim_ixs, shape):
    if isinstance(shape, md.mdarray):
        strides = shape.strides
        size = shape.size
        mdim = shape.mdim
    else:
        strides = get_strides(shape)
        size = reduce(lambda x, y: x * y, shape)
        mdim = len(shape)

    mdim_ixs = tuple(mdim_ixs)
    ndim = len(mdim_ixs)
    ixs = [0] * ndim

    for i in range(ndim):
        mdim_ix_i = mdim_ixs[i]
        ixs[i] = pair_wise_accumulate(strides, mdim_ix_i)

    return ixs


'''
M-d array slicing:
'''


def unravel_dense(*dense_ixs, arr_in, arr_out, set):
    global j
    dense_ixs = tuple(dense_ixs)
    ndim = len(dense_ixs)
    mdim = dense_ixs[0].mdim
    shape = dense_ixs[0].shape
    strides = arr_in.strides

    def recurse(ix):
        global j
        axis = shape[ix]

        if ix == 0:
            for i in range(axis):
                ix_i = 0
                for k in range(ndim):
                    ix_k = dense_ixs[k].data[j] * strides[k]
                    ix_i += ix_k
                if set:
                    arr_in.data[ix_i] = arr_out.data[j]
                else:
                    arr_out.data[j] = arr_in.data[ix_i]
                j += 1
        else:
            for i in range(axis):
                recurse(ix - 1)
    j = 0
    recurse(mdim - 1)


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
        if not isinstance(arr_i, md.mdarray):
            if arr_i == inf or arr_i == Ellipsis:
                arr_i = irange(arr.shape[i])
            else:
                arr_i = tomdarray(slc[i])

        new_shape[i] = arr_i.size
        oned = False if arr_i.mdim > 1 else oned
        slc[i] = arr_i

    return slc, new_shape, oned


def slice_array(slc, arr_in, arr_out, set=True):
    slc, new_shape, oned = expand_indicies(slc, arr_in)
    order = arr_in.order

    if oned:
        slc = dense_meshgrid(*slc)
        slc = broadcast_arrays(*slc)
    else:
        slc = broadcast_arrays(*slc)
        new_shape = slc[0].shape

    if not arr_out:
        arr_out = zeros(new_shape, order=order, dtype=arr_in.dtype)
    else:
        arr_out = tomdarray(arr_out)

    if arr_in.shape != arr_out.shape:
        arr_out = broadcast_toshape(arr_out, new_shape)

    unravel_dense(*slc, arr_in=arr_in, arr_out=arr_out, set=set)
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
