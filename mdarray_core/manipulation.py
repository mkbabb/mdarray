from functools import reduce

import numpy as np

import mdarray as md
from mdarray_core.creation import full, zeros
from mdarray_core.exceptions import IncompatibleDimensions
from mdarray_core.helper import (get_strides, pair_wise_accumulate, roll_array,
                                 swap_item)
from mdarray_core.indexing import flatten_list, make_nested_list
from mdarray_core.types import inf, nan

# __all__ = ["mdarray_iter", "concatenate", "hstack", "vstack", "dstack", "roll_axis", "pad_array",
#            "repeat", "meshgrid", "reduce_array"]


'''
Reshaping routines:
'''


def reshape(arr, new_shape):
    mdim = len(new_shape)
    new_size = reduce(lambda x, y: x*y, new_shape)

    if new_size != arr.size:
        raise IncompatibleDimensions
    else:
        arr.shape = new_shape
        arr.mdim = mdim
        arr.strides = get_strides(new_shape)


def transpose(arr, axis1=0, axis2=1):
    swap_item(arr.strides, axis1, axis2)
    swap_item(arr.shape, axis1, axis2)


def swap_axis(arr, axis1=0, axis2=1):
    transpose(arr, axis1, axis2)


def roll_axis(arr, axis, iterations=1):
    roll_array(arr.shape, axis, iterations)
    roll_array(arr.strides, axis, iterations)


def flatten(arr, order=1):
    new_mdim = arr.mdim - order
    new_shape = [0]*(arr.mdim - order)

    for i in range(new_mdim - 1):
        new_shape[i] = arr.shape[i]

    init = 1
    for i in range(order + 1):
        init *= arr.shape[i]

    new_shape[0] = init

    reshape(arr, new_shape)


def ravel_internal(ix, mdim_ix_i, strides, size, mdim):
    j = 0
    while j < mdim:
        stride = strides[mdim - (j + 1)]
        k = 1

        while True:
            stride_k = stride*k
            if stride_k >= ix:
                if stride != 1:
                    k -= 1
                stride_k = stride*k
                break
            else:
                k += 1

        ix -= stride_k
        mdim_ix_i[mdim - (j + 1)] = k
        j += 1
    return mdim_ix_i


def ravel(*ixs, shape):
    if isinstance(shape, md.mdarray):
        strides = shape.strides
        size = shape.size
        mdim = shape.mdim
    else:
        strides = get_strides(shape)
        size = reduce(lambda x, y: x*y, shape)
        mdim = len(shape)

    ixs = tuple(ixs)
    ndim = len(ixs)
    mdim_ixs = [[0]*mdim]*ndim

    for i in range(ndim):
        ix = ixs[i]
        if ix > size:
            raise IncompatibleDimensions(
                "The raveled index is too large to unravel using the provided shape!")
        mdim_ix_i = [0]*mdim
        mdim_ixs[i] = ravel_internal(ix, mdim_ix_i, strides, size, mdim)

    return mdim_ixs


def unravel(*mdim_ixs, shape):
    if isinstance(shape, md.mdarray):
        strides = shape.strides
        size = shape.size
        mdim = shape.mdim
    else:
        strides = get_strides(shape)
        size = reduce(lambda x, y: x*y, shape)
        mdim = len(shape)

    mdim_ixs = tuple(mdim_ixs)
    ndim = len(mdim_ixs)
    ixs = [0]*ndim

    for i in range(ndim):
        mdim_ix_i = mdim_ixs[i]
        ixs[i] = pair_wise_accumulate(strides, mdim_ix_i)

    return ixs


'''
End reshaping and retyping routines.
'''


'''
Concatenation and splitting routines:
'''


def concatenate(*arrs, caxis):
    global j
    arr1 = arrs[0]

    ndim = len(arrs)
    mdim = arr1.mdim

    new_shape = list(arr1.shape)
    new_shape[caxis] = 0

    axis_counters = [[0]*mdim]*ndim
    for i in range(ndim):
        arr_i = arrs[i]

        if mdim != arr_i.mdim:
            raise IncompatibleDimensions(
                "The dimensions of array one does not equal the rest!")

        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise IncompatibleDimensions(
                        "The shape of array one (disregarding caxis) does not equal the rest!")
        new_shape[caxis] += arr_i.shape[caxis]

    arr_out = zeros(shape=new_shape)

    def recurse(warr, ix):
        global j
        axis_counter = axis_counters[warr]
        arr_i = arrs[warr]

        shape = arr_i.shape
        strides = arr_i.strides
        data = arr_i.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i

                ix3 = pair_wise_accumulate(axis_counter, strides)

                try:
                    a_val = data[ix3]
                except:
                    a_val = nan

                arr_out.data[j] = a_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(warr, ix - 1)

                if ix == caxis + 1:
                    for k in range(ndim - 1):
                        recurse(k + 1, ix - 1)

    j = 0
    if caxis == mdim - 1:
        for i in range(ndim):
            recurse(i, mdim - 1)
    else:
        recurse(0, mdim - 1)

    return arr_out


def hstack(*arrs):
    return concatenate(*arrs, caxis=0)


def vstack(*arrs):
    return concatenate(*arrs, caxis=1)


def dstack(*arrs):
    return concatenate(*arrs, caxis=2)


def tile(arr, tiles):
    mdim = arr.mdim
    ndim = len(tiles)
    if ndim > mdim:
        new_shape = arr.shape + [1]*(ndim - mdim)
        arr = arr.reshape(new_shape)

    arr_i = arr
    for i in range(ndim):
        tile_i = tiles[i]
        arrs = [0]*tile_i

        for j in range(tile_i):
            arrs[j] = arr_i

        arr_i = concatenate(*arrs, caxis=i)
    return arr_i


# Implicitly a concatenation routine: uses pdim pad arrays concatenated with the main "arr" array.

def pad_array(arr, pad_width, pad_values):
    ndim = len(pad_width)
    pdim = len(pad_width[0])

    if not isinstance(pad_values, list):
        pad_values = [pad_values]*ndim

    shape = arr.shape
    new_shape = list(shape)

    for i in range(ndim):
        v = reduce(lambda x, y: x+y, pad_width[i])
        new_shape[i] += v

    arrs = [0]*(pdim + 1)
    middle = len(arrs)//2
    shape_i = list(shape)
    a_i = arr

    for i in range(ndim):
        pad_i = pad_width[i]

        for j in range(pdim):
            shape_ij = list(shape_i)
            shape_ij[i] = pad_i[j]

            a_ij = full(shape=shape_ij, fill_value=pad_values[i])
            arrs[j] = a_ij

        arrs[pdim] = a_i
        swap_item(arrs, middle, pdim)

        a_i = concatenate(*arrs, caxis=i)
        shape_i[i] = new_shape[i]

    return a_i


'''
End concatenation and splitting routines.
'''


'''
Generalised reduction routines:
'''


def reduce_array(arr, faxis, func, mode="value"):
    global j, k

    roll_axis(arr, faxis)

    mdim = arr.mdim
    new_shape = list(arr.shape)
    new_shape.pop(0)

    arr_out = zeros(shape=new_shape)
    tmp0 = [0]*arr.shape[0]
    axis_counter = [0]*mdim

    def recurse(ix):
        global j, k
        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i

                ix_i = pair_wise_accumulate(axis_counter, strides)

                try:
                    arr_val = data[ix_i]
                except:
                    arr_val = nan

                tmp0[j] = arr_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(ix - 1)

                if ix == 1:
                    j = 0
                    arr_out.data[k] = func(tmp0)
                    k += 1

    j = k = 0
    recurse(mdim - 1)
    return arr_out


'''
End generalised reduction routines.
'''


'''
Recursive iteration template for which nearly all mdarray manipulations are based off of.
'''


def mdarray_iter(arr):
    global j
    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides
    data = arr.data

    axis_counter = [0]*mdim
    arr_out = zeros(arr.shape)

    def recurse(ix):
        global j

        axis = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, strides)
                a_val = data[ix_i]
                arr_out.data[j] = a_val

        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
    return arr_out
