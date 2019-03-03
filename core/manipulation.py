from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from core.creation import full, zeros
from core.exceptions import IncompatibleDimensions
from core.helper import get_strides, roll_array, swap_item, make_mdim_shape
from core.types import inf, nan
from MultiArray import MultiArray

__all__ = ["make_nested_list",
           "reshape", "make_mdim", "transpose", "swap_axis", "roll_axis",
           "flatten", "astype",
           "concatenate", "hstack", "vstack", "dstack",
           "tile"]


'''
Reshaping routines:
'''


def reshape(arr, new_shape):
    new_shape = list(new_shape)
    mdim = len(new_shape)
    new_size = reduce(lambda x, y: x * y, new_shape)

    if new_size != arr.size:
        raise IncompatibleDimensions(
            "The desired shape is incompatible with the current array's shape.")
    else:
        arr._shape = new_shape
        arr._mdim = mdim
        arr._strides = get_strides(new_shape)


def make_mdim(arr, mdim):
    new_shape = make_mdim_shape(arr.shape, mdim)
    reshape(arr, new_shape)


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


def make_nested_list(arr: MultiArray) -> list:
    mdim = arr.mdim
    size = arr.size
    data = arr.data
    mditer = arr.iterator

    arr = []
    nests = [[] for i in range(mdim - 1)]
    for i in range(size):
        arr.append(data[mditer.index])
        next(mditer)

        for j in range(1, mdim):
            if mditer.was_advanced[j]:
                if j == 1:
                    nests[0].append(arr)
                    arr = []
                else:
                    nests[j - 1].append(nests[j - 2])
                    nests[j - 2] = []
    arr = nests[-1]
    mditer.at(0)
    return arr


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


def _confirm_concat_shape(arrs: Tuple[MultiArray, ...],
                          caxis: int,
                          mdim: int,
                          ndim: int,
                          new_shape: List[int]) -> List[int]:
    new_shape[caxis] = 0
    for i in range(ndim):
        arr_i = arrs[i]
        if mdim != arr_i.mdim:
            raise IncompatibleDimensions(
                f"The dimensions of array 1 != the dimensions of array {i}!")
        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise IncompatibleDimensions(
                        "All axes but caxis must be equivalent to concatenate the arrays.")
        new_shape[caxis] += arr_i.shape[caxis]
    return new_shape


def concatenate(*arrs: MultiArray,
                caxis: int = -1) -> MultiArray:
    arrs = tuple(arrs)
    ndim = len(arrs)
    arr1 = arrs[0]
    mdim = arr1.mdim

    if caxis < 0:
        caxis += mdim

    new_shape = _confirm_concat_shape(arrs, caxis, mdim, ndim, list(arr1.shape))
    arr_out = zeros(shape=new_shape, order=arr1.order, dtype=arr1.dtype)

    if caxis != mdim - 1:
        k = 0
        while k < arr_out.size:
            for i in range(ndim):
                for j in arrs[i].iterator:
                    if j.was_advanced[caxis + 1]:
                        j._was_advanced[caxis + 1] = False
                        break
                    else:
                        arr_out.data[k] = arrs[i].data[j.index]
                        k += 1
    else:
        k = 0
        while k < arr_out.size:
            for i in range(ndim):
                for j in arrs[i].data:
                    arr_out.data[k] = j
                    k += 1
    for i in range(ndim):
        arrs[i].iterator.at(0)
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
