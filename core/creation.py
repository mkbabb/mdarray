import types
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from core.exceptions import IncompatibleDimensions
from core.helper import flatten_list, make_mdim_shape, roll_array, swap_item
from MultiArray import MultiArray

__all__ = ["tomdarray", "tondarray",
           "zeros", "ones", "full",
           "irange", "linear_range", "log_range",
           "repeat", "ix_meshgrid", "meshgrid", "dense_meshgrid",
           "generate_broadcast_shape", "broadcast_nary", "broadcast", "broadcast_iter",
           "broadcast_toshape", ]


'''
toarray routines:

'''


def tomdarray(arr: Union[MultiArray, np.ndarray, list, tuple, Any]
              ) -> MultiArray:
    if isinstance(arr, MultiArray):
        return arr

    elif isinstance(arr, np.ndarray):
        arr_out = MultiArray(data=np.ravel(
            arr), shape=arr.shape, order=arr.order)
        return arr_out

    else:
        if isinstance(arr, list) or isinstance(arr, tuple):
            arr, mdim, shape = flatten_list(arr, order=-1)
            arr_out = MultiArray(shape=shape, data=arr)
        elif isinstance(arr, int) or isinstance(arr, float) or isinstance(arr, str):
            arr_out = MultiArray(size=1, data=[arr])
        else:
            arr = list(arr)
            arr_out = MultiArray(size=len(arr), data=arr)
        return arr_out


def tondarray(arr: Union[MultiArray, np.ndarray, list, tuple, Any]
              ) -> np.ndarray:
    if isinstance(arr, MultiArray):
        nd = np.asarray(arr.data, dtype=arr.dtype,
                        order=arr.order).reshape(arr.shape[::-1])
    else:
        nd = np.asarray(arr)
    return nd


'''
End toarray routines.
'''


'''
M-d arrays filled with pre-defined values:
'''


def zeros(shape: Optional[List[int]] = None,
          size: Optional[int] = None,
          dtype: Optional[Any] = None,
          order: Optional[str] = None) -> MultiArray:
    arr_out = MultiArray(shape=shape, size=size, order=order)
    arr_out._data = [0] * arr_out.size
    return arr_out


def ones(shape: Optional[List[int]] = None,
         size: Optional[int] = None,
         dtype: Optional[Any] = None,
         order: Optional[str] = None) -> MultiArray:
    arr_out = MultiArray(shape=shape, size=size, order=order)
    arr_out._data = [1] * arr_out.size
    return arr_out


def full(fill: Any = 0,
         shape: Optional[List[int]] = None,
         size: Optional[int] = None,
         dtype: Optional[Any] = None,
         order: Optional[str] = None) -> MultiArray:
    arr_out = MultiArray(shape=shape, size=size, order=order)
    arr_out._data = [fill] * arr_out.size
    return arr_out


'''
End M-d arrays
'''


'''
Array ranges:
'''


def irange(size: Union[int, list]) -> MultiArray:
    if isinstance(size, list):
        shape = size
        size = reduce(lambda x, y: x * y, size)
    else:
        shape = [size]
    data = [i for i in range(size)]
    arr = MultiArray(shape=shape, data=data)
    return arr


def linear_range(start: Union[int, float],
                 stop: Union[int, float],
                 size=Optional[int]) -> MultiArray:
    if not size:
        size = stop - start
    arr_out = zeros(shape=[size])

    if type(start) == int:
        step = (stop - start) // size
    else:
        step = (stop - start) / size

    i = start
    j = 0
    while j < size:
        arr_out.data[j] = i
        j += 1
        i += step
    return arr_out


def log_range(start: Union[int, float],
              stop: Union[int, float],
              base: Union[int, float],
              size=Optional[int]) -> MultiArray:
    return base**linear_range(start, stop, size)


'''
End array ranges.
'''


'''
Tiling and grid routines:
'''

# Uses the same recursive function as broadcast_bnry,
# but manipulates the data in a different way.


def _sort_axes(raxes: List[int],
               repts: List[int],
               mdim: int) -> List[int]:
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


def repeat(arr: MultiArray,
           raxes: List[int],
           repts: List[int]) -> MultiArray:
    mdim = arr.mdim
    _sort_axes(raxes, repts, mdim)
    new_shape = list(arr.shape)

    for i in range(mdim):
        rept = repts[i]
        raxis = raxes[i]
        new_shape[raxis] *= rept

    arr.repeats = repts
    arr_out = zeros(shape=new_shape, order=arr.order, dtype=arr.dtype)

    for n, i in enumerate(arr):
        arr_out.data[n] = arr.data[i.index]

    return arr_out


def meshgrid_internal(arrs: List[MultiArray],
                      _iter: bool = True) -> List[MultiArray]:
    sizes = list(map(len, arrs))
    mdim = len(arrs)

    arrs_out = [0] * mdim

    for i in range(mdim):
        slc = [1] * mdim
        slc[i] = sizes[i]
        if _iter:
            arrs_out[i] = MultiArray(shape=slc)
        else:
            arrs_out[i] = tomdarray(arrs[i]).reshape(slc)

    return arrs_out


def ix_meshgrid(*arrs: MultiArray) -> List[MultiArray]:
    return meshgrid_internal(arrs, True)


def dense_meshgrid(*arrs: MultiArray) -> List[MultiArray]:
    return meshgrid_internal(arrs, False)


def meshgrid(*arrs: MultiArray) -> List[MultiArray]:
    arrs = meshgrid_internal(arrs, False)
    return broadcast(*arrs)


'''
End tiling and grid routines.
'''


'''
Broadcasting routines:
'''


def generate_broadcast_shape(*arrs: MultiArray
                             ) -> (List[int], List[List[int]]):
    '''
    From a collection of disparately sized, but broadcastable,
    arrays, forms a common broadcast shape and set of repeats
    necessary for each array to be broadcasted to the aforesaid shape.

    Broadcasting heuristics can be described as follows:

    First, each input array is dimensionally padded to the maximal 
    input dimension.

    We then iterate through each dimension sequentially and take the maximal
    value therefrom.

    Whilst in this dimension, if any of the other input arrays have a
    axis size == 1 or == maximal_axis, they are compatible with 
    broadcasting (think of a demension size of 1 as essentially superfluous).
    The set of repeats necessary to achieve this are then created from this maximal
    axis and the new broadcast shape is created.

    If the axis size != 1 or != maximal_axis, the input arrays are not
    broadcastable, so we throw.
    '''
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


new_shape, repts = generate_broadcast_shape(MultiArray(
    shape=[5, 1, 7]), MultiArray(shape=[5, 1, 1]), MultiArray(shape=[1, 1, 7]))

print(new_shape, repts)


def broadcast_iter(*arrs: MultiArray) -> None:
    new_shape, repts = generate_broadcast_shape(*arrs)
    for i in range(len(arrs)):
        arrs[i].repeats = repts[i]
    return new_shape


def broadcast_nary(*arrs: MultiArray,
                   func: Callable[[Any], List[Any]]) -> MultiArray:
    new_shape = broadcast_iter(arrs)
    arr_out = zeros(new_shape)
    ndim = len(arrs)
    fargs = [0] * ndim

    for n, i in enumerate(zip(*arrs)):
        for m, j in enumerate(i):
            fargs[m] = arrs[m].data[j.index]
        arr_out.data[n] = func(fargs)
    return arr_out


def broadcast_toshape(arr: MultiArray,
                      shape: List[int]) -> MultiArray:
    arr_shape = MultiArray(shape=shape, order=arr.order)
    new_shape = broadcast_iter(arr, arr_shape)
    arr_out = zeros(new_shape)

    for n, i in enumerate(arr):
        arr_out.data[n] = arr.data[i.index]
    return arr_out


def broadcast(*arrs: MultiArray) -> List[MultiArray]:
    ndim = len(arrs)
    new_shape = broadcast_iter(*arrs)
    arrs_out = [zeros(new_shape) for i in range(ndim)]

    for n, i in enumerate(arrs):
        for m, j in enumerate(i):
            arrs_out[n].data[m] = arrs[n].data[j.index]
    return arrs_out
