from functools import reduce

import numpy as np

import mdarray as md
from core.exceptions import IncompatibleDimensions
from core.helper import pair_wise_accumulate, roll_array

__all__ = ["zeros", "ones", "full",
           "arange", "linear_range", "log_range",
           "repeat", "meshgrid", "dense_meshgrid",
           "diagonal", "eye",
           "generate_broadcast_shape", "broadcast_bnry", "broadcast_arrays"]

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

# Uses the same recursive function as broadcast_bfunction,
# but manipulates the data in a different way.


def repeat(arr, raxes, repts):
    global j

    if not isinstance(raxes, list):
        raxes = [raxes]
    if not isinstance(repts, list):
        repts = [repts]

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

                a_val = arr.data[ix_i]

                arr_out.data[j] = a_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                for k in range(ndim):
                    raxis = raxes[k]
                    rept = repts[k]
                    if ix == raxis or (raxis == 0 and ix == 1):
                        for l in range(rept):
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


def diagonal(arr):
    mdim = arr.mdim
    size = arr.size
    shape = arr.shape

    if mdim == 1:
        arr_out = zeros([size, size])
        col_stride = arr_out.strides[1]
        for i in range(size):
            arr_out.data[i*(col_stride + 1)] = arr.data[i]
    elif mdim == 2:
        arr_out = zeros([shape[0]])
        col_stride = arr.strides[1]
        for i in range(shape[0]):
            arr_out.data[i] = arr.data[i*(col_stride+1)]

    return arr_out


def eye(order=2):
    return diagonal(ones([order]))


'''
End tiling and grid routines.
'''


'''
Broadcasting routines:
'''


def broadcast_bnry_internal(*arrs, new_shape, raxes, repts, func):
    global j
    arrs = tuple(arrs)
    ndim = len(arrs)

    mdim = arrs[0].mdim
    arr_out = zeros(shape=new_shape)

    axis_counters = [[0]*mdim for i in range(ndim)]

    def recurse(warr, ix, flag=False):
        print(mdim, ix)
        global j
        arr = arrs[warr]
        shape = arr.shape
        strides = arr.strides
        axis = shape[ix]

        axis_counter = axis_counters[warr]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(
                    axis_counter, strides)

                if flag:
                    arr_out.data[j] = func(
                        arr_out.data[j], arr.data[ix_i])
                else:
                    arr_out.data[j] = arr.data[ix_i]
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i

                for k in range(len(raxes[warr])):
                    raxis = raxes[warr][k]
                    rept = repts[warr][k]
                    print(raxis, rept, ix)

                    if ix == raxis or (raxis == 0 and ix == 1):
                        for l in range(rept):
                            recurse(warr, ix - 1, flag)
                        break
                else:
                    recurse(warr, ix - 1, flag)

    j = 0
    recurse(0, mdim - 1, False)
    for i in range(ndim - 1):
        j = 0
        print(arr_out)
        recurse(i+1, mdim - 1, True)

    '''
    If an n-ary (n > 2) function is wished to be broadcast_bfunction to n arrays simultaneously,
    the most cost efficient way of doing so is via coroutines.
    
    Since C/C++ does not yet properly support this functionality,
    I leave it as an optional implmentation for possible later use.

    To modify the current broadcast_bfunction routine to support this,
    simply add a "yield from" statement to every recursive call above, and a "yield" statment to the "ix_i" variable.
    Then, comment out the block of code above, and comment in the block of code below.
    '''

    # casts = [recurse(i, mdim-1) for i in range(ndim)]
    # fargs = [0]*ndim
    # for i in range(arr_out.size):
    #     for j in range(ndim):
    #         ix_j = next(casts[j])
    #         arrs_j = arrs[j].data[ix_j]
    #         fargs[j] = arrs_j
    #     arr_out.data[i] = func(*fargs)

    return arr_out


def broadcast_arrays_internal(*arrs, new_shape, raxes, repts):
    global j
    arrs = tuple(arrs)
    ndim = len(arrs)

    mdim = arrs[0].mdim
    arrs_out = [zeros(shape=new_shape) for i in range(ndim)]

    axis_counters = [[0]*mdim for i in range(ndim)]

    def recurse(warr, ix):
        global j
        arr = arrs[warr]
        shape = arr.shape
        strides = arr.strides
        axis = shape[ix]

        arr_out = arrs_out[warr]

        axis_counter = axis_counters[warr]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(
                    axis_counter, strides)
                arr_out.data[j] = arr.data[ix_i]
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i

                for k in range(len(raxes[warr])):
                    raxis = raxes[warr][k]
                    rept = repts[warr][k]
                    if ix == raxis or (raxis == 0 and ix == mdim - 1):
                        for l in range(rept):
                            recurse(warr, ix - 1)
                        break
                else:
                    recurse(warr, ix - 1)

    for i in range(ndim):
        j = 0
        recurse(i, mdim - 1)

    return arrs_out


def generate_broadcast_shape(*arrs):
    arrs = tuple(arrs)
    ndim = len(arrs)

    shapes = [i.shape for i in arrs]
    mdims = [i.mdim for i in arrs]
    mdim = max(mdims)

    if mdims[0] < mdim:
        shapes[0] += [1]*(mdim - mdims[0])
    arrs[0].reshape(shapes[0])

    raxes = [[] for i in range(ndim)]
    repts = [[] for i in range(ndim)]
    new_shape = list(shapes[0])

    for i in range(mdim):
        axis_i = shapes[0][i]

        for j in range(1, ndim):
            if i == 0:
                shape_j = shapes[j]
                mdim_j = mdims[j]

                if mdim_j < mdim:
                    shapes[j] += [1]*(mdim - mdim_j)
                arrs[j].reshape(shapes[j])

            axis_j = shapes[j][i]

            if axis_i == 1 and axis_j > 1:
                axis_i = axis_j
                raxes[0].append(i)
                repts[0].append(axis_i)
            elif axis_i > 1 and axis_j == 1:
                raxes[j].append(i)
                repts[j].append(axis_i)
            elif axis_i != axis_j:
                raise IncompatibleDimensions

        new_shape[i] = axis_i

    return new_shape, raxes, repts


def broadcast_bnry(*arrs, func):
    new_shape, raxes, repts = generate_broadcast_shape(*arrs)
    print(new_shape, raxes, repts)
    print([i.shape for i in arrs])
    arr_out = broadcast_bnry_internal(
        *arrs, new_shape=new_shape, raxes=raxes, repts=repts, func=func)
    return arr_out


def broadcast_arrays(*arrs, shape=None):
    if shape:
        arr_shape = md.mdarray(shape=shape)
        new_shape, raxes, repts = generate_broadcast_shape(*arrs, arr_shape)
        arr_out = broadcast_arrays_internal(
            *arrs, arr_shape, new_shape=new_shape, raxes=raxes, repts=repts)
    else:
        new_shape, raxes, repts = generate_broadcast_shape(*arrs)
        print(new_shape, raxes, repts)
        arr_out = broadcast_arrays_internal(
            *arrs, new_shape=new_shape, raxes=raxes, repts=repts)
    return arr_out
