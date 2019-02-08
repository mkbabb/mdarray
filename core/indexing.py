from functools import reduce

import mdarray as md
from core.creation import (arange, broadcast_arrays, dense_meshgrid, tomdarray,
                           zeros)
from core.exceptions import IncompatibleDimensions
from core.helper import get_strides, pair_wise_accumulate
from core.types import inf, nan

__all__ = ["ravel", "unravel",
           "unravel_dense_get", "unravel_dense_set", "expand_indicies",
           "slice_array_get", "slice_array_set"]


'''
Raveling and unraveling indicies:
'''


def ravel_internal(ix, mdim_ix_i, strides, size, mdim):
    j = 0
    while j < mdim:
        stride = strides[mdim - (j + 1)]
        k = 1

        while True:
            stride_k = stride * k
            if stride_k >= ix:
                if stride != 1:
                    k -= 1
                stride_k = stride * k
                break
            else:
                k += 1

        ix -= stride_k
        mdim_ix_i[mdim - (j + 1)] = k
        j += 1
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
        if ix > size:
            raise IncompatibleDimensions(
                "The raveled index is too large to unravel using the provided shape!")
        mdim_ix_i = [0] * mdim
        mdim_ixs[i] = ravel_internal(ix, mdim_ix_i, strides, size, mdim)

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


def unravel_dense_get(*dense_ixs, arr, new_shape):
    global j
    dense_ixs = tuple(dense_ixs)
    ndim = len(dense_ixs)
    mdim = dense_ixs[0].mdim
    shape = dense_ixs[0].shape
    strides = dense_ixs[0].strides
    astrides = arr.strides

    arr_out = zeros(new_shape)

    axis_counter = [0] * mdim

    def recurse(ix):
        global j

        axis = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, strides)
                ix_k = 0

                for k in range(ndim):
                    ix_k += dense_ixs[k].data[ix_i] * astrides[k]

                arr_out.data[j] = arr.data[ix_k]
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
    return arr_out


def unravel_dense_set(*dense_ixs, arr1, arr2):
    global j
    dense_ixs = tuple(dense_ixs)
    ndim = len(dense_ixs)
    mdim = dense_ixs[0].mdim
    shape = dense_ixs[0].shape
    strides = dense_ixs[0].strides
    astrides = arr1.strides

    axis_counter = [0] * mdim

    def recurse(ix):
        global j

        axis = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, strides)
                ix_k = 0

                for k in range(ndim):
                    ix_k += dense_ixs[k].data[ix_i] * astrides[k]

                arr1.data[ix_k] = arr2.data[j]
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
    return arr1


def expand_indicies(slc, arr):
    slc = list(slc)
    ndim = len(slc)
    oned = True
    new_shape = [0] * ndim

    i = 0
    while i < ndim:
        arr_i = slc[i]

        if arr_i == inf or arr_i == Ellipsis:
            arr_i = arange(arr.shape[i])
        else:
            arr_i = tomdarray(slc[i])

        new_shape[i] = arr_i.size
        oned = False if arr_i.mdim > 1 else oned
        slc[i] = arr_i
        i += 1

    return slc, new_shape, oned


def slice_array_get(slc, arr):
    slc, new_shape, oned = expand_indicies(slc, arr)

    if oned:
        slc = dense_meshgrid(*slc)
        slc = broadcast_arrays(*slc)
    else:
        slc = broadcast_arrays(*slc)
        new_shape = slc[0].shape

    arr_vals = unravel_dense_get(*slc, arr=arr, new_shape=new_shape)
    return arr_vals


def slice_array_set(slc, arr1, arr2):
    slc, new_shape, oned = expand_indicies(slc, arr1)
    arr2 = tomdarray(arr2)

    if oned:
        slc = dense_meshgrid(*slc)
        slc = broadcast_arrays(*slc)
    else:
        slc = broadcast_arrays(*slc)
        new_shape = slc[0].shape

    arr_shape = zeros(shape=new_shape)
    arr2 = broadcast_arrays(arr2, arr_shape)[0]

    unravel_dense_set(*slc, arr1=arr1, arr2=arr2)
    return arr1


'''
End M-d array slicing
'''
