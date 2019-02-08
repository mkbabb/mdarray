import mdarray as md
import core


def dot(arr1, arr2):
    shape1 = arr1.shape
    shape2 = arr2.shape

    if shape1[1] != shape2[1]:
        raise core.IncompatibleDimensions
    else:
        arr_out = core.zeros([arr2.shape[0], arr1.shape[0]])
        for i in range(arr2.shape[0]):
            ix = i, core.inf
            arr_val = core.reduce_array(arr1 * (arr2[ix].T()), 0, sum)
            arr_out[i, core.inf] = arr_val
    return arr_out


def norm(arr, metric=2):
    arr_out = core.reduce_array(arr**metric, 0, sum)
    return arr_out**(1 / metric)
