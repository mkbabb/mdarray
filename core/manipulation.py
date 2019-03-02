from functools import reduce

import numpy as np

import MultiArray as ma
from core.creation import full, zeros
from core.exceptions import IncompatibleDimensions
from core.helper import (get_strides, roll_array,
                         swap_item)
from core.types import inf, nan

__all__ = ["make_nested_list",
           "reshape", "transpose", "swap_axis", "roll_axis",
           "flatten", "astype",
           "concatenate", "hstack", "vstack", "dstack",
           "tile", "mdarray_iter"]


'''
Reshaping routines:
'''


def reshape(arr, new_shape):
    new_shape = list(new_shape)
    mdim = len(new_shape)
    new_size = reduce(lambda x, y: x * y, new_shape)

    if new_size != arr.size:
        raise IncompatibleDimensions("The desired shape is incompatible with the current array's shape.")
    else:
        arr._shape = new_shape
        arr._mdim = mdim
        arr._strides = get_strides(new_shape)


def transpose(arr, axis1=0, axis2=1):
    mdim = arr.mdim
    if axis1 < 0:
        axis1 += mdim
    if axis2 < 0:
        axis2 += mdim

    maxis = max(axis1, axis2)
    if maxis > mdim - 1:
        paxis = maxis - (mdim - 1)
        reshape(arr, arr.shape + [1] * paxis)

    swap_item(arr.strides, axis1, axis2)
    swap_item(arr.shape, axis1, axis2)


def swap_axis(arr, axis1=0, axis2=1):
    transpose(arr, axis1, axis2)


def roll_axis(arr, axis, iterations=1):
    mdim = arr.mdim
    if axis < 0:
        axis += mdim
    roll_array(arr.shape, axis, iterations)
    roll_array(arr.strides, axis, iterations)


def flatten(arr, order=-1):
    mdim = arr.mdim

    if order < 0:
        order += mdim
    elif order == 0:
        return arr

    new_mdim = arr.mdim - order
    new_shape = [0] * (arr.mdim - order)

    for i in range(new_mdim):
        new_shape[i] = arr.shape[i]

    red = 1
    for i in range(new_mdim - 1, mdim):
        red *= arr.shape[i]

    new_shape[-1] = red
    reshape(arr, new_shape)


def make_nested_list(arr):
    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides
    data = arr.data
    axis_counter = [0] * mdim

    def recurse(ix):
        axis = shape[ix]
        tmp = [0] * axis

        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)
                arr_val = data[ix_i]
                tmp[i] = arr_val
        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                tmp[i] = recurse(ix - 1)
        return tmp
    return recurse(mdim - 1)


def astype(arr, dtype):
    shape = list(arr.shape)
    mdim = arr.mdim
    size = arr.size

    if dtype != arr.dtype:
        if dtype == complex and mdim > 1:
            if shape[0] % 2 == 0:
                shape[0] //= 2
                arr_out = zeros(shape=shape, order=arr.order, dtype=complex)
                j = 0
                for i in range(0, size, 2):
                    arr_out.data[j] = arr.data[i] + 1j * arr.data[i + 1]
                    j += 1
                arr = arr_out
            else:
                raise IncompatibleDimensions
        elif arr.dtype == complex:
            shape[0] *= 2
            arr_out = zeros(shape=shape, order=arr.order, dtype=dtype)
            j = 0
            for i in range(0, arr_out.size, 2):
                arr_out.data[i] = arr.data[j].real
                arr_out.data[i + 1] = arr.data[j].imag
                j += 1
            arr = arr_out
        else:
            for i in range(size):
                arr.data[i] = dtype(arr.data[i])
    return arr


'''
End reshaping and retyping routines.
'''


'''
Concatenation and splitting routines:
'''


def concatenate(*arrs, caxis):
    global j, k

    arr1 = arrs[0]

    ndim = len(arrs)
    mdim = arr1.mdim

    if caxis < 0:
        caxis += mdim

    new_shape = list(arr1.shape)
    new_shape[caxis] = 0

    for i in range(ndim):
        arr_i = arrs[i]

        if mdim != arr_i.mdim:
            raise IncompatibleDimensions(
                "The dimensions of array one do not equal the dimensions of array two!")

        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise IncompatibleDimensions(
                        "All axes but caxis must be equivalent to concatenate the arrays.")
        new_shape[caxis] += arr_i.shape[caxis]

    arr_out = zeros(shape=new_shape, order=arr1.order, dtype=arr1.dtype)
    axis_counter = [0] * mdim
    strides = arr_out.strides

    def recurse(warr, ix):
        global j, k
        arr = arrs[warr]
        shape = arr.shape
        axis = shape[ix]

        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter) + k * strides[caxis]
                arr_val = arr.data[j]
                arr_out.data[ix_i] = arr_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                recurse(warr, ix - 1)
    j = k = 0
    for i in range(ndim):
        recurse(i, mdim - 1)
        j = 0
        k += arrs[i].shape[caxis]

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
        new_shape = arr.shape + [1] * (ndim - mdim)
        arr = arr.reshape(new_shape)

    arr_i = arr
    for i in range(ndim):
        tile_i = tiles[i]
        arrs = [0] * tile_i

        for j in range(tile_i):
            arrs[j] = arr_i

        arr_i = concatenate(*arrs, caxis=i)
    return arr_i


'''
End concatenation and splitting routines.
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
    axis_counter = [0] * mdim

    def recurse(ix):
        global j
        axis = shape[ix]
        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)
                arr_val = data[ix_i]

        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
