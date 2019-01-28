from functools import reduce

import numpy as np

from mdarray import arange, mdarray, tomdarray, zeros
from mdarray_formatting import pad_array_fmt
from mdarray_helper import (get_strides, pair_wise_accumulate, swap_item,
                            update_dict)
from mdarray_indexing import gslice, iter_gslice, make_nested
from mdarray_types import IncompatibleDimensions, inf, mdarray_inquery, nan


def roll_array(arr, pos, iterations=1):
    mdim = len(arr)

    if pos == mdim - 1:
        return arr

    def recurse(ix):
        swap_item(arr, pos, ix)
        ix -= 1
        if ix == pos:
            return
        else:
            recurse(ix)

    for i in range(iterations):
        recurse(mdim - 1)

    return arr


def roll_axis(arr, axis, iterations=1):
    roll_array(arr.shape, axis, iterations)
    roll_array(arr.strides, axis, iterations)


def accumulate(arr, faxis, func):
    global j, k

    roll_axis(arr, faxis)

    mdim = arr.mdim
    new_shape = list(arr.shape)
    new_shape.pop(-1)

    arr_out = zeros(shape=new_shape)

    tmp0 = [0]*arr_out.size
    axis_counter = [0]*mdim

    def recurse(ix):
        global j, k
        shape = arr.shape
        strides = arr.strides
        data = arr.data
        pos = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == 1:
            for i in range(pos):
                axis_counter[mdim - 1] = i

                ix_i = pair_wise_accumulate(axis_counter, strides)

                try:
                    arr_val = data[ix_i]
                except:
                    arr_val = nan

                tmp0[j] = arr_val
                j += 1
        else:
            for i in range(pos):
                axis_counter[ix] = i
                recurse(ix + 1)

                if ix == mdim - 2:
                    j = 0
                    arr_out.data[k] = func(tmp0)
                    k += 1
        return j

    j = k = 0
    recurse(0)
    return arr_out


def repeat(arr, rept, raxis):
    data = arr.data
    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides

    new_shape = list(shape)
    new_shape[raxis] *= rept

    axis_counter = [0]*mdim
    arr_out = [0]*arr.size*rept

    raxis_s = 1 if mdim - 1 != raxis else rept

    def recurse(axis_counter, ix, j):
        pos = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == 1:
            for i in range(pos):
                for k in range(raxis_s):
                    axis_counter[mdim - 1] = i
                    ix3 = pair_wise_accumulate(axis_counter, strides)

                    try:
                        a_val = data[ix3]
                    except:
                        a_val = nan

                    arr_out[j] = a_val
                    j += 1
        else:
            for i in range(pos):
                axis_counter[ix] = i
                if ix == raxis:
                    for k in range(rept):
                        j = recurse(axis_counter, ix + 1, j)
                else:
                    j = recurse(axis_counter, ix + 1, j)

        return j

    recurse(axis_counter, 0, 0)
    return tomdarray(arr_out).reshape(new_shape)


def meshgrid(*arrs):
    arrs = tuple(arrs)
    lens = list(map(len, arrs))
    mdim = len(arrs)

    size = 1
    for i in range(mdim):
        size *= lens[i]

    arr_out = []
    for n, i in enumerate(arrs):
        slc = [1]*mdim
        slc[n] = lens[n]
        arr_i = tomdarray(i).reshape(slc)
        for m, j in enumerate(lens):
            if m != n:
                arr_i = repeat(arr_i, j, raxis=m)
        arr_out.append(arr_i)

    return tuple(arr_out)


def concatenate(*arrs, caxis):
    arrs = list(arrs)
    arr1 = arrs[0]

    ndim = len(arrs)
    mdim = arr1.mdim

    new_shape = list(arr1.shape)
    new_size = arr1.size

    ixs = [[0]*mdim]*ndim
    for i in range(ndim - 1):
        arr_i = arrs[i]

        if mdim != arr_i.mdim:
            raise IncompatibleDimensions(
                "The dimensions of array one does not equal the rest!")

        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise IncompatibleDimensions(
                        "The shape of array one (disregarding caxis) does not equal the rest!")

        new_size += arr_i.size
        new_shape[caxis] += arr_i.shape[caxis]

    arr_out = [0]*(new_size)

    def recurse(warr, ix, j):
        axis_counter = ixs[warr]
        arr_i = arrs[warr]

        shape = arr_i.shape
        strides = arr_i.strides
        data = arr_i.data
        pos = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == 1:
            for i in range(pos):
                axis_counter[mdim - 1] = i

                ix3 = pair_wise_accumulate(axis_counter, strides)

                try:
                    a_val = data[ix3]
                except:
                    a_val = nan

                arr_out[j] = a_val
                j += 1
        else:
            for i in range(pos):
                axis_counter[ix] = i
                j = recurse(warr, ix + 1, j)

                if ix == caxis - 1:
                    for k in range(ndim - 1):
                        j = recurse(k + 1, ix + 1, j)
        return j

    if caxis == 0:
        j = 0
        for i in range(ndim):
            j = recurse(i, 0, j)
    else:
        recurse(0, 0, 0)

    return tomdarray(arr_out).reshape(new_shape)


# shape1 = [5, 5, 2]
# size1 = reduce(lambda x, y: x*y, shape1)
#
# arr1 = arange(size1).reshape(shape1)
#
# print(arr1)
#
# shape2 = [5, 5, 2]
# size2 = reduce(lambda x, y: x*y, shape2)
#
# arr2 = arange(size2).reshape(shape2) - 222222
# print(arr2)
#
# arr_out = concatenate(arr1, arr2, arr2, arr2, arr1, caxis=2)
# print(arr_out)


shape1 = [5, 4, 2, 6]
size1 = reduce(lambda x, y: x*y, shape1)

arr1 = arange(size1).reshape(shape1)

faxis = 3

np_arr = np.asarray(arr1.data).reshape(shape1)


v = accumulate(arr1, faxis, sum)
print(v)
print(np_arr.sum(faxis))
