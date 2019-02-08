import mdarray as md
import core


def dot(arr1, arr2):
    shape1 = arr1.shape
    shape2 = arr2.shape
    if shape1[1] != shape2[1]:
        raise core.IncompatibleDimensions


    
    arr_out = core.reduce_array(arr1 * (arr2.T()), 0, sum)
    return arr_out.T()


def norm(arr, metric=2):
    arr_out = core.reduce_array(arr**metric, 0, sum)
    return arr_out**(1 / metric)
