from math import log, log2

import numpy as np

import mdarray as md
from mdarray_core.exceptions import IncompatibleDimensions
from mdarray_core.helper import (get_strides, pair_wise_accumulate, roll_array,
                                 swap_item)
from mdarray_core.indexing import flatten_list, make_nested_list
from mdarray_core.types import inf, nan


'''
M-d arrays filled with pre-defined values:
'''


def zeros(shape=None, **kwargs):
    arr_out = md.mdarray(shape=shape, **kwargs)
    arr_out.data = [0]*arr_out.size
    return arr_out


def ones(shape=None, **kwargs):
    arr_out = md.mdarray(shape=shape, **kwargs)
    arr_out.data = [1]*arr_out.size
    return arr_out


def full(shape=None, fill_value=0, **kwargs):
    arr_out = md.mdarray(shape=shape, **kwargs)
    arr_out.data = [fill_value]*arr_out.size
    return arr_out


'''
End M-d arrays
'''


'''
Array ranges:
'''


def arange(size):
    data = [i for i in range(size)]
    return md.mdarray(size=size, data=data)


def linear_range(start, stop, size):
    arr_out = zeros(shape=[size])

    step = (stop - start)/size

    i = start
    j = 0
    while j < size:
        arr_out.data[j] = i
        j += 1
        i += step
    return arr_out


def log_range(start, stop, size, base=10):
    return base**linear_range(start, stop, size)


'''
End array ranges
'''

'''
Tiling and grid routines:
'''


def repeat(arr, raxis, rept):
    global j
    mdim = arr.mdim

    new_shape = list(arr.shape)
    new_shape[raxis] *= rept

    arr_out = zeros(shape=new_shape)
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


def meshgrid_internal(*arrs, dense=False):
    arrs = tuple(arrs)
    sizes = list(map(len, arrs))
    mdim = len(arrs)

    arr_out = []
    for i in range(mdim):
        slc = [1]*mdim
        slc[i] = sizes[i]
        arr_i = md.tomdarray(arrs[i]).reshape(slc)

        if not dense:
            for j in range(mdim):
                if j != i:
                    arr_i = repeat(arr_i, j, sizes[j])
        arr_out.append(arr_i)

    return tuple(arr_out)


def dense_meshgrid(*arrs):
    return meshgrid_internal(*arrs, dense=True)


def meshgrid(*arrs):
    return meshgrid_internal(*arrs, dense=False)


'''
End tiling and grid routines.
'''
