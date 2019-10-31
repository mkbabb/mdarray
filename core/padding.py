from core.creation import full, zeros
from core.manipulation import concatenate
from core.reduction import reduce_array

__all__ = ["pad_array",
           "pad_median", "pad_wrap", "pad_reflect", "pad_constant"]
'''
Implicitly a concatenation routine:
Uses pdim pad arrays concatenated with the main "arr" array.
'''


def pad_array(arr, pad_width, pad_func):
    '''
    parameters:
        arr: mdarray
        Input array

        pad_width: list, tuple or mdarray
        A 2-d list or tuple, of which the first axis must be equal to 2.
        This beacuse each axis is padded only from the left or right on each iteration.

        pad_func: function
            Padding function that dictates how the input array is padded with values.
            See the padding function section for more information.
    '''
    mdim = arr.mdim
    arr_i = arr

    for i in range(-1, -mdim - 1, -1):
        pad_i = pad_width[i]
        shape_i = list(arr_i.shape)

        pad_left = []
        pad_right = []

        for j in range(2):
            pad_ij = pad_i[j]
            for k in range(pad_ij):
                shape_i[i] = k + 1
                arr_ijk = pad_func(arr=arr_i, axis=i, shape=shape_i, side=j, pad=k)
                if j == 0:
                    pad_left.append(arr_ijk)
                else:
                    pad_right.append(arr_ijk)

        pad_left.append(arr_i)
        arrs = pad_left + pad_right
        arr_i = concatenate(*arrs, caxis=i)

    return arr_i


'''
Padding functions utilized in pad_array.

A generalised padding function should be of the form:

    pad_func(arr, axis, shape, side, pad_i)

parameters:
    arr: mdarray
        Input array for which the padding function acts on.

    axis: int
        The axis of which the output is appended to.

    shape: mdarray shape; list
        The required shape of the output.

    side: int
        An integer value that is either: 0, representing the padding of the left side,
        or 1, representing the padding of the right side.

    pad_i: int
        The ith location of the pad array to be appended.

    returns:
        An mdarray that must be a first order reduction along the given axis.
        Example, if shape = [3, 3, 2] and axis = -1, the output shape must equal [3, 3, 1];
        a first order reduction along the -1 axis.
'''


def pad_median(arr, axis, shape, side, pad):
    func = lambda x: round(sum(x) / len(x))
    return reduce_array(arr, axis, func)


def pad_constant(arr, axis, shape, side, pad):
    return full(shape, fill_value=99)


def pad_reflect(arr, axis, shape, side, pad):
    if side == 0:
        func = lambda x: x[0]
    elif side == 1:
        func = lambda x: x[-1]
    return reduce_array(arr, axis, func)


def pad_wrap(arr, axis, shape, side, pad):
    if pad > 0:
        pad = pad % (arr.shape[axis])
    if side == 0:
        func = lambda x: x[-(pad + 1)]
    elif side == 1:
        func = lambda x: x[pad]
    return reduce_array(arr, axis, func)
