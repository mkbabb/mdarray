import operator
from functools import reduce
from typing import *

import numpy as np


import core.creation
import core.exceptions
import core.helper

from MultiArray import MultiArray


__all__ = ["make_nested_list",
           "reshape", "make_mdim", "transpose", "swap_axis", "roll_axis",
           "flatten",
           "concatenate", "hstack", "vstack", "dstack",
           "tile"]


'''
Reshaping routines:
'''


def reshape(arr: MultiArray,
            new_shape: List[int]):
    new_shape = list(new_shape)
    mdim = len(new_shape)
    new_size = reduce(lambda x, y: x * y, new_shape)

    if new_size != arr.size:
        raise core.exceptions.IncompatibleDimensions(
            "The desired shape is incompatible with the current array's shape.")
    else:
        arr._shape = new_shape
        arr._mdim = mdim
        arr._strides = core.helper. get_strides(new_shape)

    arr._stride_shape = core.helper.pair_wise(arr._shape,
                                              arr._strides,
                                              operator.mul)

    arr._axis_counter = [0] * arr._mdim
    arr._was_advanced = [False] * arr._mdim

    arr._rept_counter = [0] * arr._mdim
    arr._repeats = [0] * arr._mdim

    arr._pos = 0
    arr._index = 0


def make_mdim():
    pass


def transpose(arr: MultiArray,
              axis1: int = 0,
              axis2: int = 1):
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


def swap_axis(arr: MultiArray,
              axis1: int = 0,
              axis2: int = 1):
    transpose(arr, axis1, axis2)


def roll_axis(arr: MultiArray,
              axis: int,
              iterations: int = 1):
    mdim = arr.mdim
    if axis < 0:
        axis += mdim
    roll_array(arr.shape, axis, iterations)
    roll_array(arr.strides, axis, iterations)


def flatten(arr: MultiArray,
            order: int = -1):
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
    tmp = []
    nests = [[] for i in range(arr.mdim - 1)]

    for i in range(arr.size):
        tmp.append(arr.data[arr.index])
        next(arr)

        for j in range(1, arr.mdim):
            if arr.was_advanced[j]:
                if j == 1:
                    nests[0].append(tmp)
                    tmp = []
                else:
                    nests[j - 1].append(nests[j - 2])
                    nests[j - 2] = []
    arr.at(0)
    return nests[-1]


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
            raise ValueError(
                f"The dimensions of array 1 != the dimensions of array {i}!")

        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise ValueError(
                        "All axes but caxis must be equivalent to concatenate the arrays.")

        new_shape[caxis] += arr_i.shape[caxis]

    return new_shape


def concatenate(*arrs: MultiArray,
                caxis: int = -1) -> MultiArray:
    arrs = tuple(arrs)
    new_shape = _confirm_concat_shape(
        arrs, caxis, arrs[0].mdim, len(arrs), list(arrs[0].shape))

    arr_out = core.creation.zeros(shape=new_shape)

    caxis = -1 if caxis >= arr_out.mdim - 1 else caxis

    j = 0
    while (j < arr_out.size):
        for arr in arrs:
            while (not arr.was_advanced[caxis + 1]
                   and arr.index < arr.size):
                arr_out.data[j] = arr.data[arr.index]
                j += 1
                next(arr)
            arr.was_advanced[caxis + 1] = False

    for i in range(len(arrs)):
        arrs[i].at(0)

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
