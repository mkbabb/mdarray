from __future__ import annotations

import math
import operator
import random
import sys
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import core


def generate_broadcast_shape(arrs: List[Union[multiArray, multiArrayIter]]
                             ) -> (List[int], List[int]):
    ndim = len(arrs)

    shapes = [i.shape for i in arrs]
    mdims = [i.mdim for i in arrs]
    mdim = max(mdims)

    for i in range(ndim):
        core.make_mdim(shapes[i], mdim)

    repts = [[1] * mdim for i in range(ndim)]
    new_shape = [0] * mdim

    for i in range(mdim):
        axis_i = shapes[0][i]
        j = 1
        while j < ndim:
            axis_j = shapes[j][i]
            if axis_i == 1 and axis_j > 1:
                axis_i = axis_j
                repts[0][i] = axis_i
                j -= 1
            elif axis_i > 1 and axis_j == 1:
                repts[j][i] = axis_i
            elif axis_i != axis_j:
                raise ValueError
            j += 1
        new_shape[i] = axis_i
    return new_shape, repts


def broadcast_iter(iters: List[multiArrayIter]) -> List[int]:
    new_shape, repts = generate_broadcast_shape(iters)
    for i in range(len(iters)):
        iters[i].repeats = repts[i]
    return new_shape


def print_array2(maiter,
                 data,
                 sep: Optional[str] = ", ",
                 formatter=None
                 ) -> str:
    if not formatter:
        formatter = lambda x, y: f"{x[y]}"
    if not sep:
        sep = ", "

    mdim = maiter.mdim
    size = maiter.size

    s = ""
    strings = [""] * (mdim - 1)
    for i in range(size):
        s += formatter(data, maiter.index)
        next(maiter)
        s += sep if not maiter.was_advanced[1] else ''

        ix = 0
        for j in range(1, mdim):
            if maiter.was_advanced[j]:
                if j == 1:
                    strings[0] += f"[{s}]"
                    s = ""
                else:
                    strings[j - 1] += f"[{strings[j - 2]}]"
                    strings[j - 2] = ""
                ix += 1
        if ix > 0 and i != size - 1:
            new_line = "\n" * ix
            hanging_indent = " " * (mdim - ix)
            strings[ix - 1] += (sep.strip() + new_line + hanging_indent)

    s = f"[{strings[-1]}]"
    maiter.at(0)
    return s


def unravel_dense(dense_ixs: List[list, multiArray],
                  dense_iters: List[multiArrayIter],
                  arr_in: multiArray,
                  arr_out: Optional[multiArray] = None,
                  setter: bool = False) -> multiArray:
    strides = arr_in.strides
    for n, i in enumerate(zip(*dense_iters)):
        ix_i = 0
        for m, j in enumerate(i):
            print(j.repeats)
            ix_i += dense_ixs[m][j.index] * strides[m]

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
        if not isinstance(arr_i, multiArray):
            if arr_i == float("inf") or arr_i == Ellipsis:
                arr_i = core.irange(arr.shape[i])
            else:
                arr_i = core.tomdarray(slc[i])

        new_shape[i] = arr_i.size
        oned = False if arr_i.mdim > 1 else oned
        slc[i] = arr_i

    return slc, new_shape, oned


def slice_array(slc, arr_in, arr_out, setter=True):
    slc, new_shape, oned = expand_indicies(slc, arr_in)
    order = arr_in.order

    if oned:
        slc = core.ix_meshgrid(*slc)
        slc = core.broadcast_iter(*slc)
    else:
        slc = core.broadcast_iter(*slc)
        new_shape = slc[0].shape

    if not arr_out:
        arr_out = core.zeros(new_shape, order=order, dtype=arr_in.dtype)
    else:
        arr_out = core.tomdarray(arr_out)

    if arr_in.shape != arr_out.shape:
        arr_out = core.broadcast_toshape(arr_out, new_shape)
    unravel_dense(slc, arr_in, arr_out, setter)
    return arr_out


def dot2(arr1, arr2):
    if arr1.shape[-1] != arr2.shape[0]:
        raise ValueError

    for i in range(arr2.shape[0]):
        print(i)


def make_mditer(arrs):
    print(arrs)
    iters = [0] * len(arrs)
    for n, i in enumerate(arrs):
        iters[n] = multiArrayIter(i)
    return iters


# arr1 = multiArray([4, 4], [1] * 16)
# arr2 = multiArray([4, 4], [99] * 16)
# arr_out = multiArray([1, 4], [0] * 4)

# slc, new_shape, oned = expand_indicies([[0], [0, 1, 2, 3]], arr1)


def normalize_index(mdim, ix):
    if ix > mdim:
        ix = mdim - 1
    elif ix < 0:
        if ix < -mdim:
            ix = (ix % mdim) + mdim
        else:
            ix += mdim
    return ix


def grab_axis(arr_in, arr_out=None, axis=0, ix=0, slc=None):
    mdim = arr_in.mdim
    shape = arr_in.shape
    axis = normalize_index(mdim, axis)
    ix = normalize_index(shape[axis], ix)

    if isinstance(slc, type(None)):
        slc = [0] * mdim

    slc[axis] = [0]
    new_shape = list(shape)
    for i in range(mdim):
        if i != axis:
            if slc[i] != shape[i]:
                slc[i] = [j for j in range(shape[i])]

    if isinstance(arr_out, type(None)):
        arr_out = core.zeros(new_shape)

    iters = make_mditer(slc)

    unravel_dense(slc, iters, arr_in, arr_out, False)
    print(arr_out)
    return slc


s1 = [5, 4, 2, 1]
s2 = [1, 4, 2, 1]
s3 = [5, 1, 1, 1]
s4 = [1, 1, 1, 1]
s5 = [5, 4, 1, 1]
