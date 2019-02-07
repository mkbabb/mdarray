from functools import reduce

from core.helper import pair_wise_accumulate
from core.types import inf, nan

__all__ = ["expand_dims", "expand_slice_array",
           "remove_extraneous_dims", "flatten_list", "make_nested_list"]


'''
Generalised slicing of arrays
'''


class _gslice(object):
    def __init__(self, slice_array, a_inqry):
        self.slice_array = slice_array

        self.mdim = a_inqry.mdim
        self.strides = a_inqry.strides
        self.shape = a_inqry.shape

        self.arg_axis = 0

        if len(self.slice_array) == 0:
            self.slice_array = [0]

        mdim = len(self.slice_array)

        if mdim < self.mdim:
            self.slice_array += [nan] * (self.mdim - mdim)
        elif mdim > self.mdim:
            self.slice_array = self.slice_array[:self.mdim]

    def get_slice(self):
        new_shape = []
        new_strides = []
        new_size = 1

        for n, i in enumerate(self.slice_array):
            if i != nan:
                tmp = i * self.strides[n]
                self.arg_axis += tmp
                self.slice_array[n] = tmp
                self.mdim -= 1
            else:
                tmp = self.shape[n]
                new_size *= tmp
                new_shape += [tmp]
                new_strides += [self.strides[n]]

        if len(new_shape) == 0:
            self.mdim = 1
            self.shape = [1]
            self.strides = [1]
            self.size = 1
        else:
            self.shape = new_shape
            self.strides = new_strides
            self.size = new_size

        return self

    def __repr__(self):
        return str(self.slice_array)


class gslice(object):
    def __init__(self, slice_array, arr):
        tmp0 = []
        tmp1 = []
        new_shape = [0] * arr.mdim

        for n, i in enumerate(slice_array):
            if i == inf:
                slice_array[n] = [j for j in range(arr.shape[n])]

        for n, i in enumerate(slice_array):
            if isinstance(i, gslice):
                tmp0 += [i]
            else:
                tmp1 += [i]
                if len(slice_array) == arr.mdim:
                    pass

        new_shape = [i for i in new_shape if i != 0]

        self.shape = new_shape if len(new_shape) <= a_inqry.mdim else [1]

        tmp1 = remove_extraneous_dims(tmp1)
        self.slice_array = make_iter_list(tmp1)

        for n, i in enumerate(self.slice_array):
            self.slice_array[n] = _gslice(i, a_inqry).get_slice()

        for i in tmp0:
            self.slice_array += [i.slice_array[0]]

    def __repr__(self):
        if len(self.slice_array) == 1:
            return str(self.slice_array[0])
        return str(self.slice_array)


def expand_slice_array(slice_array, mdim, strides=None):
    ndim = len(slice_array)

    if strides:
        pad_length = strides[ndim - 1] if mdim != 0 else 1
    else:
        pad_length = 1

    broadcast_length = len(slice_array[0])
    out_array = [[0] * mdim] * (broadcast_length * pad_length)

    print(broadcast_length, pad_length)

    tmp0 = [0] * mdim
    l = 0
    for i in range(broadcast_length):
        for j in range(ndim):
            tmp0[j] = slice_array[j][i]

        if pad_length == 1:
            out_array[i] = list(tmp0)
        else:
            for k in range(pad_length):
                tmp0[mdim - 1] = k
                out_array[l] = list(tmp0)
                l += 1

    return out_array


def expand_dims(slice_array, arr):
    ndim = len(slice_array)
    mdim = arr.mdim

    broadcast_length = 0
    lens = 0
    for i in range(ndim):
        arr_i = slice_array[i]
        try:
            ndim_i = len(arr_i)
        except TypeError:
            if arr_i == inf or arr_i == Ellipsis:
                slice_array[i] = [j for j in range(arr.shape[i])]
                ndim_i = arr.shape[i]
            else:
                slice_array[i] = [arr_i]
                ndim_i = 1

        lens += ndim_i
        broadcast_length = ndim_i if ndim_i > broadcast_length else broadcast_length

    broadcast_length = mdim if broadcast_length > mdim else broadcast_length

    if lens // ndim == ndim:
        return slice_array

    tmp0 = [0] * broadcast_length
    for i in range(ndim):
        ndim_i = len(slice_array[i])

        ndim_i = ndim_i if ndim_i < mdim else mdim
        pad_length = broadcast_length - ndim_i

        j = 0
        while j < ndim_i:
            tmp0[j] = slice_array[i][j]
            j += 1

        if pad_length > 0:
            while j < ndim_i + pad_length:
                tmp0[j] = slice_array[i][ndim_i - 1]
                j += 1

        slice_array[i] = list(tmp0)

    return slice_array


'''
Iteration over an array by a given gslice. Used for any type of indexing into an m-d array.
'''


def iter_gslice(arr, gslice_array, size):
    data = arr.data
    a_out = [0] * size

    def recurse(g, axis_counter, ix, j):
        axis = g.shape[ix]
        remaining_axes = g.mdim - ix

        if remaining_axes == 1:
            for i in range(axis):
                axis_counter[g.mdim - 1] = i
                ix_i = pair_wise_accumulate(
                    axis_counter, g.strides) + g.arg_axis

                arr_val = data[ix_i]
                a_out[j] = arr_val
                j += 1

        else:
            for i in range(axis):
                axis_counter[ix] = i
                j = recurse(g, axis_counter, ix + 1, j)
        return j

    j = 0
    for i in gslice_array.slice_array:
        axis_counter = [0] * i.mdim
        ix = 0

        j = recurse(i, axis_counter, ix, j)

    return a_out


'''
Python list manipulations:

	remove_extraneous dims: removes arr given python list's dimensions that are empty, 
							recursively iterating inward until an axis is found to be non-empty.
	
	flatten: flattens arr multi-dimensional python list by arr given order.
	
	make_nested: creates arr multi-dimensional python list from arr 1-d contiguous m-d array.
'''


def remove_extraneous_dims(arr):
    def recurse(arr):
        if len(arr) == 1:
            try:
                arr = recurse(arr[0])
                return arr
            except IndexError:
                return arr
        else:
            return arr

    return recurse(arr)


def flatten_list(arr, order=1):
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


def make_nested_list(arr):
    mdim = arr.mdim
    axis_counter = [0] * mdim

    def recurse(ix):
        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        tmp = [0] * axis

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):

                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, strides)

                try:
                    arr_val = data[ix_i]
                except:
                    arr_val = nan

                tmp[i] = arr_val

        else:
            for i in range(axis):
                axis_counter[ix] = i
                tmp[i] = recurse(ix - 1)

        return tmp

    arr_out = recurse(mdim - 1)
    return arr_out


def make_iter_list(slice_array):
    if len(slice_array) == 0:
        return slice_array

    array_out = []

    def recurse(slice_array, array_out):
        flag = False
        recursive_flag = False
        tmp0 = []

        for n, i in enumerate(slice_array):
            if isinstance(i, list) and not recursive_flag:
                for j in slice_array[n]:
                    tmp1 = [k for m, k in enumerate(slice_array) if m != n]
                    tmp1.insert(n, j)

                    array_out = recurse(tmp1, array_out)
                    recursive_flag = True
                flag = True

            else:
                tmp0 += [i]

        if not flag:
            array_out += [tmp0]

        return array_out

    return recurse(slice_array, array_out)
