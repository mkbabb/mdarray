from __future__ import annotations

import timeit
import typing
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import core.depreciated as depreciated
from core import (IncompatibleDimensions, generate_broadcast_shape,
                  inner_product, irange, meshgrid, swap_item, tomdarray,
                  tondarray, zeros)
from multiArray import multiArray, multiArrayIter




def meshgrid_iter(arrs):
    sizes = list(map(len, arrs))
    mdim = len(arrs)

    arrs = [0] * mdim
    for i in range(mdim):
        slc = [1] * mdim
        slc[i] = sizes[i]
        arrs[i] = multiArray(shape=slc)
    return arrs


def swap_item_axis(arr: multiArray,
                   axis: int,
                   ix1: Union[int, List[int]],
                   ix2: Union[int, List[int]]) -> None:
    if ix1 == ix2:
        return
    strides = arr.strides
    stride = strides[axis]
    data = arr.data
    mditer = arr.iterator

    pos = mditer.pos
    if isinstance(ix1, list):
        ix1 = int(inner_product(ix1, strides))
    else:
        ix1 *= stride
    if isinstance(ix2, list):
        ix2 = int(inner_product(ix2, strides))
    else:
        ix2 *= stride

    ix2 -= ix1
    mditer.at(ix1)
    while True:
        ix = mditer.index
        swap_item(data, ix, ix + ix2)
        next(mditer)
        if mditer.was_advanced[axis]:
            break
    mditer.at(pos)


def partition(seq, key, axis, left, right):
    pix = left
    pivot = key(seq, axis, right)

    for i in range(left, right):
        seq_i = key(seq, axis, i)
        eq = (seq_i <= pivot)
        if eq and (pix != i):
            swap_item_axis(seq, axis, pix, i)
            pix += 1
    if pix != right:
        swap_item_axis(seq, axis, pix, right)
    return pix


def quicksort(seq, key, axis, left, right):
    if left < right:
        pix = partition(seq, key, axis, left, right)
        quicksort(seq, key, axis, left, pix - 1)
        quicksort(seq, key, axis, pix + 1, right)


def grab(arr: multiArray,
         buff: list,
         axis: int,
         func: Callable[[list], list] = None,
         count: int = 1) -> list:
    if not func:
        func = lambda x: x
    size = arr.size
    data = arr.data
    mditer = arr.iterator

    if axis < 0:
        axis += arr.mdim

    _count = 0
    for i in range(size):
        buff[i] = data[mditer.index]
        next(mditer)

        if mditer.was_advanced[axis] or axis == 0:
            _count += 1
            if _count == count:
                fbuff = func(buff)
                return fbuff
    return buff





# arr1 = irange([5, 1, 2])
# repeats1 = [2, 2, 2]

# arr_out = _cc.repeat(arr1, [0, 1, 2], repeats1)
# print(arr_out)

# arr1.iterator.repeats = repeats1

# for i in arr1.iterator:
#     print(i.index)

# raxes = [i for i in range(arr1.mdim)]
# base_arr1 = cc.repeat(arr1, raxes, repeats1)
# print(base_arr1)
