from __future__ import annotations

import timeit
import typing
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

from core import (IncompatibleDimensions, generate_broadcast_shape,
                  inner_product, swap_item, zeros, irange)
from MultiArray import MultiArray, MultiArrayIter


def broadcast_iter(arrs):
    new_shape, repts = generate_broadcast_shape(*arrs)
    for i in range(len(arrs)):
        arrs[i].iterator.repeats = repts[i]
    return new_shape


def broadcast_nary(arrs, func):
    new_shape = broadcast_iter(arrs)
    arr_out = zeros(new_shape)
    ndim = len(arrs)
    fargs = [0] * ndim

    for n, i in enumerate(zip(*arrs)):
        for m, j in enumerate(i):
            fargs[m] = arrs[m].data[j.index]
        arr_out.data[n] = func(fargs)
    return arr_out


def broadcast_toshape_iter(arr: MultiArray,
                           shape: List[int]) -> MultiArrayIter:
    arr_shape = MultiArray(shape=shape, order=arr.order, dtype=arr.dtype)
    new_shape, repts = generate_broadcast_shape(arr, arr_shape)
    arr.iterator.repeats = repts[0]
    return arr


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


def concatenate_iter(arrs: MultiArray,
                     caxis: int = -1) -> MultiArrayIter:
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


def print_iter(arr: MultiArray,
               sep: Optional[str] = ", ",
               formatter: Optional[Callable[[str], str]] = None
               ) -> str:
    if not formatter:
        formatter = lambda x: f"{x}"
    if not sep:
        sep = ", "

    mdim = arr.mdim
    size = arr.size
    data = arr.data
    mditer = arr.iterator

    s = ""
    strings = [""] * (mdim - 1)
    for i in range(size):
        s += formatter(data[mditer.index])
        next(mditer)
        s += sep if not mditer.was_advanced[1] else ''

        ix = 0
        for j in range(1, mdim):
            if mditer.was_advanced[j]:
                if j == 1:
                    strings[0] += f"[{s}]"
                    s = ""
                else:
                    strings[j - 1] += f"[{strings[j - 2]}]"
                    strings[j - 2] = ""
                ix += 1

        if ix > 0 and i != size - 1:
            strings[ix - 1] += sep
            strings[ix - 1] += "\n" * ix
            strings[ix - 1] += " " * ix if ix < mdim - 1 else ""
            strings[ix - 1] += " " if ix > 0 else ""

    s = f"[{strings[-1]}]"
    mditer.at(0)
    return s


def unravel_dense_iter(dense_ixs: list,
                       arr_in: MultiArray,
                       arr_out: Optional[MultiArray] = None,
                       setter: bool = False) -> MultiArray:
    ndim = len(dense_ixs)
    dense_iters = broadcast_iter(dense_ixs)
    if not arr_out:
        arr_out = zeros(dense_iters[0].shape)

    strides = arr_in.strides
    for n, i in enumerate(zip(*dense_iters)):
        ix_i = 0
        for m, j in enumerate(i):
            ix_i += j.arr.data[j.index] * strides[m]
        if setter:
            arr_in.data[ix_i] = arr_out.data[n]
        else:
            arr_out.data[n] = arr_in.data[ix_i]
    return arr_out


def meshgrid_iter(arrs):
    sizes = list(map(len, arrs))
    mdim = len(arrs)

    arrs = [0] * mdim
    for i in range(mdim):
        slc = [1] * mdim
        slc[i] = sizes[i]
        arrs[i] = MultiArray(shape=slc)
    return arrs


def swap_item_axis(mditer: MultiArray,
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


def grab(arr: MultiArray,
         buff: list,
         axis: int,
         func: Callable[[list], list] = None,
         count: int = 1) -> list:
    if not func:
        func = lambda x: x

    mdim = arr.mdim
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


arr = irange([5, 5, 4, 2])
