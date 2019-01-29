from functools import reduce

import numpy as np

from mdarray import arange, full, mdarray, ones, tomdarray, tondarray, zeros
from mdarray_formatting import pad_array_fmt
from mdarray_helper import (get_strides, pair_wise_accumulate, swap_item,
                            update_dict)
from mdarray_indexing import gslice, iter_gslice, make_nested
from mdarray_types import IncompatibleDimensions, inf, mdarray_inquery, nan


def roll_array(arr, axis, iterations=1):
    mdim = len(arr)

    if axis == mdim - 1:
        return arr

    def recurse(ix):
        swap_item(arr, axis, ix)
        ix -= 1
        if ix == axis:
            return
        else:
            recurse(ix)

    for i in range(iterations):
        recurse(mdim - 1)

    return arr


def roll_axis(arr, axis, iterations=1):
    roll_array(arr.shape, axis, iterations)
    roll_array(arr.strides, axis, iterations)


def iter_axis(arr, faxis, func):
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
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == 1:
            for i in range(axis):
                axis_counter[mdim - 1] = i

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
                recurse(ix + 1)

                if ix == mdim - 2:
                    j = 0
                    arr_out.data[k] = func(tmp0)
                    k += 1
        return j

    j = k = 0
    recurse(0)
    return arr_out


def reduce_array(arr, faxis, func, mode="value"):
    global j, k

    roll_axis(arr, faxis)

    mdim = arr.mdim
    new_shape = list(arr.shape)
    new_shape.pop(-1)

    arr_out = zeros(shape=new_shape)
    tmp0 = [0]*arr.shape[-1]
    axis_counter = [0]*mdim

    def recurse(ix):
        global j, k
        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == 1:
            for i in range(axis):
                axis_counter[mdim - 1] = i

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
                recurse(ix + 1)

                if ix == mdim - 2:
                    j = 0
                    arr_out.data[k] = func(tmp0)
                    k += 1

    j = k = 0
    recurse(0)
    return arr_out


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

    for i in range(1, ndim):
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

    pad_i = pad_width[0]
    for j in range(pdim):
        shape_ij = list(shape_i)
        shape_ij[0] = pad_i[j]
        arrs[j] = full(shape=shape_ij, fill_value=pad_values[0])

    arrs[pdim] = a_i
    swap_item(arrs, middle, pdim)
    a_i = concatenate(*arrs, caxis=0)
    return a_i


def repeat(arr, raxis, rept):
    global j
    j = 0
    mdim = arr.mdim

    new_shape = list(arr.shape)
    new_shape[raxis] *= rept

    arr_out = zeros(shape=new_shape)
    axis_counter = [0]*mdim

    raxis_s = 1 if mdim - 1 != raxis else rept

    def recurse(ix):
        global j

        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == 1:
            for i in range(axis):
                for k in range(raxis_s):
                    axis_counter[mdim - 1] = i
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
                if ix == raxis:
                    for k in range(rept):
                        recurse(ix + 1)
                else:
                    recurse(ix + 1)

        return j

    recurse(0)
    return arr_out


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
                arr_i = repeat(arr_i, m, j)
        arr_out.append(arr_i)

    return tuple(arr_out)


def concatenate(*arrs, caxis):
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

    print(new_shape)
    arr_out = zeros(shape=new_shape)

    def recurse(warr, ix, j):
        axis_counter = axis_counters[warr]
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

                arr_out.data[j] = a_val
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

    return arr_out


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

def func(arr):
    ndim = len(arr)
    mx = 0
    pos = 0
    for i in range(ndim):
        arr_i = arr[i]
        if arr_i > mx:
            mx = arr_i
            pos = i
    return pos


# shape1 = [5, 3, 2]
# size1 = reduce(lambda x, y: x*y, shape1)

# arr1 = arange(size1).reshape(shape1)
# np_arr = tondarray(arr1)

# print(arr1)
# raxis = 0
# v = reduce_array(arr1, raxis, func, "index")



# ixs = tondarray(v)

# ixs = np.argmax(np_arr, axis=raxis)
# print(ixs)
# i, j, k = np.unravel_index(ixs, np_arr.shape)


# print(np_arr[ixs[0]])

# faxis = 3

# v = pad(arr1, [[2, 2], [2, 2]], 99)
# print(v)

# pad1 = (1, 1)
# pad2 = (1, 1)
# pad3 = (1, 1)

# v1 = full(shape=[1, 1, 3], fill_value=99)
# v2 = full(shape=[1, 4, 1], fill_value=99)
# v3 = full(shape=[1, 4, 5], fill_value=99)

# v7 = concatenate(v1, arr1, v1, caxis=1)
# v8 = concatenate(v2, v7, v2, caxis=2)
# v9 = concatenate(v3, v8, v3, caxis=0)
# pdd = pad_array(arr1, (pad1, pad2, pad3), 99)
# print(pdd)

# b1 = full(shape=[1, 2, 3], fill_value=99)
# b2 = full(shape=[1, 1, 3], fill_value=99)


# aa = concatenate(b2, arr1, b1, caxis=1)
# print(aa)
# print(v9)


# print(v3, v3.shape)
# print(v9, v9.shape)

# print('\n\n')
# np_arr = np.asarray(arr1.data).reshape(shape1)
# pd = np.pad(np_arr, (pad1, pad2, pad3), 'constant', constant_values=(99,))
# print(pd)
# print(pd.shape)

# print(v)
# raxis = 0
# print(np.repeat(np_arr, 2, raxis))


# v = repeat(arr1, raxis, 2)
# print(v)

# v = reduce_array(arr1, faxis, sum)
# print(v)
# print(np_arr.sum(faxis))
