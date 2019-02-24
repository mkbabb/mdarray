from __future__ import annotations

import operator
import random
import typing
from functools import reduce
from MDIter import MDIter

import numpy as np

import mdarray as md
from core import *


def get_ret_shaped2(buff, arr, new_shape, axis, keepdims):
    buff = tomdarray(buff)
    if buff.size > 1:
        if not keepdims:
            new_shape.pop(axis)
            new_shape = buff.shape + new_shape
    else:
        if keepdims:
            new_shape[axis] = 1
        else:
            new_shape.pop(axis)
    arr_out = zeros(new_shape)
    return arr_out


def _insert_into_flattened2(buff, arr_out, j):
    if isinstance(buff, md.mdarray):
        buff = buff.data

    if isinstance(buff, list):
        for n, i in enumerate(buff):
            arr_out.data[n + j] = i
        j += len(buff)
    else:
        arr_out.data[j] = buff
        j += 1
    return j


def reduce_iter(arr: md.mdarray, faxis: int,
                func: typing.Callable[[list], list],
                keepdims: bool = False) -> md.mdarray:
    mdim = arr.mdim
    roll_axis(arr, faxis)
    mditer = MDIter(arr)
    shape = arr.shape
    new_shape = list(shape)
    buff = [0] * shape[0]

    j = 0
    for i in range(shape[faxis]):
        fbuff = mditer.grapple(buff, 1, func)
        if i == 0:
            arr_out = get_ret_shaped2(fbuff, arr, new_shape,
                                      faxis, keepdims)
        j = _insert_into_flattened2(fbuff, arr_out, j)

    roll_axis(arr, faxis, mdim - 1)
    return arr_out


def repeat_iter(arr: md.mdarray, raxes: list, repts: list) -> md.mdarray:
    ndim = len(raxes)
    strides = arr.strides
    shape = arr.shape
    mditer = MDIter(arr)

    def recurse(mditer, ix, start, end):
        if ix == 0:
            for i in range(repts[ix]):
                mditer.at(start)
                for k in range(strides[ix]):
                    print(mditer.index)
                    next(mditer)
                mditer.at(end)
        else:
            for i in range(repts[ix]):
                for k in range(shape[ix]):
                    recurse(mditer, ix - 1, start + k, start + i)
                    pass

    repts = [1, 2]
    start = 0
    for i in mditer:
        if i.was_advanced[-1]:
            end = i.pos
            recurse(mditer, arr.mdim - 1, start, end)
            mditer.at(end)
            start = end
        print('----')

        # for raxis, rept in zip(raxes, repts):
        #     if i.was_advanced[raxis]:
        #         end = mditer.pos
        #         for k in range(rept):
        #             if raxis > 0:
        #                 mditer.at(start)
        #             for l in range(strides[raxis]):
        #                 print(mditer.index)
        #                 next(mditer)
        #             mditer.at(end)
        #         start = end


def swap_iter(mditer: MDIter, axis: int, ix1: list, ix2: list) -> None:
    size: int = mditer.arr.strides[axis]
    data = mditer.arr.data
    buff = [0] * size

    mditer.at(ix1)
    ixs1 = mditer.grapple(buff, axis)
    mditer.at(ix2)

    for i in range(size):
        next(mditer)
        ixs2 = mditer.index
        swap_item(data, ixs1[i], ixs2)


# arr = irange([5, 5])
# print(arr)
# narr = tondarray(arr)
# mditer = MDIter(arr)
# for i in mditer:
#     print(i.index)


# repeat_iter(arr, [0], [2])


# swap_iter(mditer, 1, [0, 1], [0, 2])

# mditer.at([0, 1])
# for i in mditer:
#     print(i.index, i.pos)
# mditer.at(2)
# for i in mditer:
#     print(i.index, i.pos)
# print(arr)

# mditer.at([0, 2])
# print(mditer.axis_counter)

# buff = [0] * arr.shape[0]

# buff = mditer.grapple(buff, 1)
# print(buff)


# print(np.repeat(narr, 2, 1))


# t = reduce_iter(arr, 0, sum, True)
# print(t.shape)
# print(t)
