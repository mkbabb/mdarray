import core
from MultiArray import MultiArray
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


arr1 = core.irange(25)
arr2 = core.irange(5).reshape([1, 5])


def _confirm_concat_shape(arrs: Tuple[MultiArray, ...],
                          caxis: int,
                          mdim: int,
                          ndim: int,
                          new_shape: List[int]) -> List[int]:
    new_shape[caxis] = 0

    for i in range(ndim):
        arr_i = arrs[i]
        if mdim != arr_i.mdim:
            raise ValueError(
                f"The dimensions of array 1 != the dimensions of array {i}!")

        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise ValueError(
                        "All axes but caxis must be equivalent to concatenate the arrays.")

        new_shape[caxis] += arr_i.shape[caxis]

    return new_shape


arrs = [arr1, arr2]
caxis = 0

new_shape = _confirm_concat_shape(
    arrs, caxis, arr1.mdim, len(arrs), list(arr1.shape))

arr_out = core.zeros(shape=new_shape)


data = []

j = 0
while (j < arr_out.size):
    for arr in arrs:
        while (True):
            next(arr)
            if (not arr.was_advanced[caxis + 1]):
                j += 1
            else:
                break

print(data)
print(j)
