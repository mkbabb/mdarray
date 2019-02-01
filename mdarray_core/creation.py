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
End array ranges.
'''

'''
Tiling and grid routines:
'''


def repeat(arr, raxes, repts):
    global j
    shape = arr.shape
    strides = arr.strides
    mdim = arr.mdim
    
    ndim = len(raxes)
    new_shape = list(arr.shape)

    for i in range(ndim):
        raxis = raxes[i]
        rept = repts[i]
        new_shape[raxis] *= rept

    arr_out = zeros(shape=new_shape)
    axis_counter = [0]*mdim

    def recurse(ix):
        global j
        axis = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, strides)

                try:
                    a_val = arr.data[ix_i]
                except:
                    a_val = nan

                arr_out.data[j] = a_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                repeated = False
                for k in range(ndim):
                    raxis = raxes[k]
                    rept = repts[k]
                    if ix == raxis or (raxis == 0 and ix == 1):
                        for l in range(rept):
                            recurse(ix - 1)
                        repeated = True
                if not repeated:
                    recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
    return arr_out

# def repeat(arr, raxis, rept):
#     global j
#     mdim = arr.mdim

#     new_shape = list(arr.shape)
#     new_shape[raxis] *= rept

#     arr_out = zeros(shape=new_shape)
#     axis_counter = [0]*mdim

#     raxis_s = 1 if 0 != raxis else rept

#     def recurse(ix):
#         global j

#         shape = arr.shape
#         strides = arr.strides
#         data = arr.data
#         axis = shape[ix]

#         remaining_axes = mdim - ix

#         if remaining_axes == mdim:
#             for i in range(axis):
#                 for k in range(raxis_s):
#                     axis_counter[0] = i
#                     ix3 = pair_wise_accumulate(axis_counter, strides)

#                     try:
#                         a_val = data[ix3]
#                     except:
#                         a_val = nan

#                     arr_out.data[j] = a_val
#                     j += 1
#         else:
#             for i in range(axis):
#                 axis_counter[ix] = i
#                 if ix == raxis:
#                     for k in range(rept):
#                         recurse(ix - 1)
#                 else:
#                     recurse(ix - 1)
#     j = 0
#     recurse(mdim - 1)
#     return arr_out


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


'''
Broadcasting routines:
'''


def broadcast_internal(arr1, arr2, raxes, repts, arr_out, func, flag=False):
    global j
    mdim = arr1.mdim
    ndim = len(raxes)
    axis_counter = [0]*mdim

    def recurse(ix):
        global j
        shape = arr1.shape
        strides = arr1.strides
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, strides)

                if not flag:
                    arr_out.data[j] = ix_i
                else:
                    ix_j = arr_out.data[j]
                    arr_out.data[j] = func(arr2.data[ix_j], arr1.data[ix_i])
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                repeated = False
                for k in range(ndim):
                    raxis = raxes[k]
                    rept = repts[k]
                    if ix == raxis or (raxis == 0 and ix == 1):
                        for l in range(rept):
                            recurse(ix - 1)
                        repeated = True
                        break
                if not repeated:
                    recurse(ix - 1)
    j = 0
    recurse(mdim - 1)


def broadcast(arr1, arr2, func):
    mdim1 = arr1.mdim
    mdim2 = arr2.mdim
    shape1 = arr1.shape
    shape2 = arr2.shape

    mdim = mdim1

    if mdim1 > mdim2:
        shape2 = shape2 + [1]*(mdim1 - mdim2)
        arr2.reshape(shape2)
    elif mdim1 < mdim2:
        shape1 = shape1 + [1]*(mdim2 - mdim1)
        arr1.reshape(shape1)
        mdim = mdim2

    raxes1 = []
    raxes2 = []
    repts1 = []
    repts2 = []
    new_shape = list(shape1)

    for i in range(mdim):
        axis1_i = shape1[i]
        axis2_i = shape2[i]
        if axis1_i == 1 and axis2_i > 1:
            repts1.append(axis2_i)
            raxes1.append(i)
            new_shape[i] = axis2_i
        elif axis1_i > 1 and axis2_i == 1:
            repts2.append(axis1_i)
            raxes2.append(i)
            new_shape[i] = axis1_i
        elif axis1_i != axis2_i:
            raise IncompatibleDimensions

    arr_out = zeros(shape=new_shape)
    broadcast_internal(arr1, arr2, raxes1, repts1, arr_out, func, False)
    broadcast_internal(arr2, arr1, raxes2, repts2, arr_out, func, True)
    return arr_out


'''
Much more straightforward implmentation of generalised broadcasting,
but is orders of magnitude more memory-expensive.
Don't use!
'''


def broadcast_copy(arr1, arr2):
    mdim1 = arr1.mdim
    mdim2 = arr2.mdim
    shape1 = arr1.shape
    shape2 = arr2.shape

    mdim = mdim1

    if mdim1 > mdim2:
        arr2.reshape(shape2 + [1]*(mdim1 - mdim2))
    elif mdim1 < mdim2:
        arr1.reshape(shape1 + [1]*(mdim2 - mdim1))
        mdim = mdim2

    shape1 = arr1.shape
    shape2 = arr2.shape

    repts1 = [0]*mdim
    repts2 = [0]*mdim

    for i in range(mdim):
        axis1_i = shape1[i]
        axis2_i = shape2[i]
        if axis1_i == 1 and axis2_i > 1:
            arr1 = repeat(arr1, i, axis2_i)
        elif axis1_i > 1 and axis2_i == 1:
            arr_out = repeat(arr_out, i, axis1_i)
        elif axis1_i != axis2_i:
            raise IncompatibleDimensions
    return arr1, arr_out
