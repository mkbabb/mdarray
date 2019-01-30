from functools import reduce

import mdarray as md
from mdarray_core.exceptions import IncompatibleDimensions
from mdarray_core.helper import pair_wise_accumulate, roll_array, swap_item
from mdarray_core.types import inf, nan

__all__ = ["mdarray_iter", "concatenate", "hstack", "vstack", "dstack", "roll_axis", "pad_array",
           "repeat", "meshgrid", "reduce_array", "flatten", "make_nested"]


'''
Recursive iteration template for which nearly all mdarray manipulations are based off of.
'''


def mdarray_iter(arr):
    global j
    mdim = arr.mdim
    axis_counter = [0]*mdim
    arr_out = md.zeros(arr.shape)

    def recurse(ix):
        global j
        shape = arr.shape
        strides = arr.strides
        data = arr.data
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

    arr_out = md.zeros(shape=new_shape)

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


def roll_axis(arr, axis, iterations=1):
    roll_array(arr.shape, axis, iterations)
    roll_array(arr.strides, axis, iterations)


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

            a_ij = md.full(shape=shape_ij, fill_value=pad_values[i])
            arrs[j] = a_ij

        arrs[pdim] = a_i
        swap_item(arrs, middle, pdim)

        a_i = concatenate(*arrs, caxis=i)
        shape_i[i] = new_shape[i]

    return a_i


def repeat(arr, raxis, rept):
    global j
    mdim = arr.mdim

    new_shape = list(arr.shape)
    new_shape[raxis] *= rept

    arr_out = md.zeros(shape=new_shape)
    axis_counter = [0]*mdim

    raxis_s = 1 if 0 != raxis else rept

    def recurse(ix):
        global j

        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                for k in range(raxis_s):
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
                if ix == raxis:
                    for k in range(rept):
                        recurse(ix - 1)
                else:
                    recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
    return arr_out


def meshgrid(*arrs):
    arrs = tuple(arrs)
    sizes = list(map(len, arrs))
    mdim = len(arrs)

    arr_out = []
    for i in range(mdim):
        slc = [1]*mdim
        slc[i] = sizes[i]
        arr_i = md.tomdarray(arrs[i]).reshape(slc)
        for j in range(mdim):
            if j != i:
                arr_i = repeat(arr_i, j, sizes[j])
        arr_out.append(arr_i)

    return tuple(arr_out)


def reduce_array(arr, faxis, func, mode="value"):
    global j, k

    roll_axis(arr, faxis)

    mdim = arr.mdim
    new_shape = list(arr.shape)
    new_shape.pop(0)

    arr_out = md.zeros(shape=new_shape)
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


def flatten(arr, order=1):
    global shape, dim_counter
    shape = [len(arr)]
    dim_counter = 0

    def recurse(arr):
        global shape, dim_counter
        ndim = len(arr)

        tmp = []
        dim_counter = 0

        for i in range(ndim):
            a_i = arr[i]

            if isinstance(a_i, list):
                tmp0 = recurse(a_i)
                M = len(a_i)

                if len(shape) <= dim_counter + 1:
                    shape.insert(0, M)

                dim_counter += 1
                tmp += [tmp0] if dim_counter <= order else tmp0
            else:
                tmp += [a_i]

        return tmp

    flt = recurse(arr)
    return flt, dim_counter, shape


def make_nested(arr):
    mdim = arr.mdim
    axis_counter = [0]*mdim

    def recurse(ix):
        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        tmp = [0]*axis

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):

                axis_counter[0] = i
                ix3 = pair_wise_accumulate(axis_counter, strides)

                try:
                    a_val = data[ix3]
                except:
                    a_val = nan

                tmp[i] = a_val

        else:
            for i in range(axis):
                axis_counter[ix] = i
                tmp[i] = recurse(ix - 1)

        return tmp

    arr_out = recurse(mdim - 1)
    return arr_out
