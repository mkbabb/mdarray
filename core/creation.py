from functools import reduce

import numpy as np

import mdarray as md
from core.exceptions import IncompatibleDimensions
from core.helper import (flatten_list, pair_wise_accumulate, roll_array,
                         swap_item)

__all__ = ["tomdarray", "tondarray",
           "zeros", "ones", "full",
           "arange", "linear_range", "log_range",
           "repeat", "meshgrid", "dense_meshgrid", "sort_raxes", "make_mdim",
           "diagonal", "eye",
           "generate_broadcast_shape", "broadcast_bnry", "broadcast_arrays"]


'''
toarray routines:
'''


def tomdarray(arr):
    if isinstance(arr, md.mdarray):
        return arr
    elif isinstance(arr, np.ndarray):
        arr_out = md.mdarray(data=np.ravel(arr), shape=arr.shape, dtype=arr.dtype, order=arr.order)
        return arr_out
    else:
        if isinstance(arr, list) or isinstance(arr, tuple):
            arr, mdim, shape = flatten_list(arr, order=-1)
            shape = [1]
            arr_out = md.mdarray(shape=shape, data=arr)
        elif isinstance(arr, dict):
            arr_out = [[i, j] for i, j in arr.items()]
        else:
            arr_out = [arr]

        return tomdarray(arr_out)


def tondarray(arr):
    nd = np.asarray(arr.data, dtype=arr.dtype, order=arr.order).reshape(arr.shape[::-1])
    return nd


'''
End toarray routines.
'''


'''
M-d arrays filled with pre-defined values:
'''


def zeros(shape=None, **kwargs):
    arr_out = md.mdarray(shape=shape, **kwargs)
    arr_out.data = [0] * arr_out.size
    return arr_out


def ones(shape=None, **kwargs):
    arr_out = md.mdarray(shape=shape, **kwargs)
    arr_out.data = [1] * arr_out.size
    return arr_out


def full(shape=None, fill_value=0, **kwargs):
    arr_out = md.mdarray(shape=shape, **kwargs)
    arr_out.data = [fill_value] * arr_out.size
    return arr_out


'''
End M-d arrays
'''


'''
Array ranges:
'''


def arange(size):
    if isinstance(size, list):
        shape = size
        size = reduce(lambda x, y: x * y, size)
    else:
        shape = [size]
    data = [i for i in range(size)]

    return md.mdarray(size=size, data=data).reshape(shape)


def linear_range(start, stop, size):
    arr_out = zeros(shape=[size])

    step = (stop - start) / size

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

# Uses the same recursive function as broadcast_bnry,
# but manipulates the data in a different way.


def make_mdim(arr, ndim):
    mdim = arr.mdim
    shape = arr.shape

    if mdim < ndim:
        arr.reshape(shape + [1] * (ndim - mdim))
    elif mdim > ndim:
        arr.flatten(order=mdim - ndim)


def sort_raxes(raxes, repts, mdim):
    ndim = len(raxes)
    if mdim > ndim:
        pad = [1] * (mdim - ndim)
        raxes += pad
        repts += pad

    def recurse(ix):
        for i in range(ix, mdim):
            raxis = raxes[i]
            rept = repts[i]
            if raxis != i and rept != 1:
                swap_item(raxes, i, raxis)
                swap_item(repts, i, raxis)
                if raxis > i:
                    recurse(i + 1)
    recurse(0)


def repeat(arr, raxes, repts):
    global j

    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides

    sort_raxes(raxes, repts, mdim)

    new_shape = list(arr.shape)

    for i in range(mdim):
        rept = repts[i]
        raxis = raxes[i]
        new_shape[raxis] *= rept

    arr_out = zeros(shape=new_shape)
    axis_counter = [0] * mdim

    def recurse(ix):
        global j
        axis = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for k in range(repts[0]):
                for i in range(axis):
                    axis_counter[0] = i
                    ix_i = pair_wise_accumulate(axis_counter, strides)

                    arr_val = arr.data[ix_i]

                    arr_out.data[j] = arr_val
                    j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i

                for k in range(1, mdim):
                    rept = repts[k]
                    if ix == k:
                        for l in range(rept):
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
        slc = [1] * mdim
        slc[i] = sizes[i]

        arr_i = tomdarray(arrs[i]).reshape(slc)

        if not dense:
            for j in range(mdim):
                if j != i:
                    arr_i = repeat(arr_i, [j], [sizes[j]])
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
            arr_out.data[i * (col_stride + 1)] = arr.data[i]
    elif mdim == 2:
        arr_out = zeros([shape[0]])
        col_stride = arr.strides[1]
        for i in range(shape[0]):
            arr_out.data[i] = arr.data[i * (col_stride + 1)]

    return arr_out


def eye(order=2):
    return diagonal(ones([order]))


'''
End tiling and grid routines.
'''


'''
Broadcasting routines:
'''


def broadcast_bnry_internal(*arrs, new_shape, repts, func):
    global j
    arrs = tuple(arrs)
    ndim = len(arrs)

    mdim = arrs[0].mdim
    arr_out = zeros(shape=new_shape)

    axis_counters = [[0] * mdim for i in range(ndim)]

    def recurse(warr, ix, flag):
        global j
        arr = arrs[warr]
        shape = arr.shape
        strides = arr.strides
        axis = shape[ix]
        repts_j = repts[warr]

        axis_counter = axis_counters[warr]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                for k in range(repts_j[0]):
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

                for k in range(1, mdim):
                    rept = repts_j[k]
                    if ix == k:
                        for l in range(rept):
                            recurse(warr, ix - 1, flag)

    for i in range(ndim):
        j = 0
        flag = True if i != 0 else False
        recurse(i, mdim - 1, flag)

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


def generate_broadcast_shape(*arrs):
    arrs = tuple(arrs)
    ndim = len(arrs)

    shapes = [i.shape for i in arrs]
    mdims = [i.mdim for i in arrs]
    mdim = max(mdims)

    for i in range(ndim):
        if mdims[i] < mdim:
            shapes[i] += [1] * (mdim - mdims[i])
        arrs[i].reshape(shapes[i])

    repts = [[1] * mdim for i in range(ndim)]
    new_shape = list(shapes[0])

    for i in range(mdim):
        axis_i = shapes[0][i]
        j = 1
        while j < ndim:
            axis_j = shapes[j][i]

            if axis_i == 1 and axis_j > 1:
                axis_i = axis_j
                repts[0][i] = axis_i
                j = 0
            elif axis_i > 1 and axis_j == 1:
                repts[j][i] = axis_i
            elif axis_i != axis_j:
                raise IncompatibleDimensions
            j += 1

        new_shape[i] = axis_i

    return new_shape, repts


def broadcast_bnry(*arrs, func):
    new_shape, repts = generate_broadcast_shape(*arrs)
    arr_out = broadcast_bnry_internal(
        *arrs, new_shape=new_shape, repts=repts, func=func)
    return arr_out


def broadcast_arrays(*arrs):
    arrs = list(arrs)
    ndim = len(arrs)

    new_shape, repts = generate_broadcast_shape(*arrs)

    raxes = [i for i in range(len(new_shape))]
    for i in range(ndim):
        arrs[i] = repeat(arrs[i], raxes, repts[i])

    return arrs
