from core.exceptions import IncompatibleDimensions
from core.creation import zeros


__all__ = ["make_nested_list",
           "concatenate", "hstack", "vstack", "dstack",
           "mdarray_iter"]


def make_nested_list(arr):
    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides
    data = arr.data
    axis_counter = [0] * mdim

    def recurse(ix):
        axis = shape[ix]
        tmp = [0] * axis

        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)
                arr_val = data[ix_i]
                tmp[i] = arr_val
        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                tmp[i] = recurse(ix - 1)
        return tmp
    return recurse(mdim - 1)


'''
Concatenation and splitting routines:
'''


def concatenate(*arrs, caxis):
    global j, k

    arr1 = arrs[0]

    ndim = len(arrs)
    mdim = arr1.mdim

    if caxis < 0:
        caxis += mdim

    new_shape = list(arr1.shape)
    new_shape[caxis] = 0

    for i in range(ndim):
        arr_i = arrs[i]

        if mdim != arr_i.mdim:
            raise IncompatibleDimensions(
                "The dimensions of array one do not equal the dimensions of array two!")

        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise IncompatibleDimensions(
                        "All axes but caxis must be equivalent to concatenate the arrays.")
        new_shape[caxis] += arr_i.shape[caxis]

    arr_out = zeros(shape=new_shape, order=arr1.order, dtype=arr1.dtype)
    axis_counter = [0] * mdim
    strides = arr_out.strides

    def recurse(warr, ix):
        global j, k
        arr = arrs[warr]
        shape = arr.shape
        axis = shape[ix]

        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter) + k * strides[caxis]
                arr_val = arr.data[j]
                arr_out.data[ix_i] = arr_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                recurse(warr, ix - 1)
    j = k = 0
    for i in range(ndim):
        recurse(i, mdim - 1)
        j = 0
        k += arrs[i].shape[caxis]

    return arr_out


def hstack(*arrs):
    return concatenate(*arrs, caxis=0)


def vstack(*arrs):
    return concatenate(*arrs, caxis=1)


def dstack(*arrs):
    return concatenate(*arrs, caxis=2)


'''
End concatenation and splitting routines.
'''


'''
Recursive iteration template for which nearly all mdarray manipulations are based off of.
'''


def mdarray_iter(arr):
    global j
    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides
    data = arr.data
    axis_counter = [0] * mdim

    def recurse(ix):
        global j
        axis = shape[ix]
        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)
                arr_val = data[ix_i]

        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                recurse(ix - 1)
    j = 0
    recurse(mdim - 1)
