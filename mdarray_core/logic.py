from functools import reduce

import numpy as np

import mdarray as md
from mdarray_core.creation import full, zeros
from mdarray_core.exceptions import IncompatibleDimensions
from mdarray_core.helper import (get_strides, pair_wise_accumulate, roll_array,
                                 swap_item)
from mdarray_core.indexing import flatten_list, make_nested_list
from mdarray_core.manipulation import roll_axis
from mdarray_core.types import inf, nan

__all__ = ["mask"]


def mask(arr, predicate):
    mdim = arr.mdim
    axis_counter = [0]*mdim

    if not predicate:
        def predicate(x): return True

    def recurse(ix):
        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i

                ix_i = pair_wise_accumulate(axis_counter, strides)

                try:
                    arr_val = data[ix_i]
                except:
                    arr_val = nan

                data[ix_i] = predicate(arr_val)
        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(ix - 1)

    recurse(mdim - 1)
    return arr

