import operator
from inspect import signature

import numpy as np

import mdarray as md
from core.creation import zeros, make_mdim
from core.helper import pair_wise, pair_wise_accumulate
from core.manipulation import roll_axis

__all__ = ["reductor", "inner_product",
           "reduce_array"]


'''
Generalised reduction of a 1-d array.
The reductor class can either apply a reduction or an accumulation
across a 1-d array. Functions of an airty >= 2 are accepted for both routines.
'''


class reductor(object):
    def __init__(self, op=None, ix=0, stride=1):
        self.op = op
        self.ix = ix
        self.stride = stride
        if self.op:
            self.nargs = len(signature(self.op).parameters)

    def reduce(self, arr):
        size = len(arr)
        t = (size // self.stride) % self.nargs

        if (t != 0) & (t != self.nargs - 1):
            raise ValueError

        i = self.ix
        out = arr[0]
        args = [0] * (self.nargs)
        self.stride *= (self.nargs - 1)

        while i < size - 1:
            args[0] = out

            for j in range(1, self.nargs):
                args[j] = arr[i + j]

            out = self.op(*args)
            i += self.stride
        return out

    def accumulate(self, arr):
        size = len(arr)
        t = (size // self.stride) % self.nargs

        if (t != 0) & (t != self.nargs - 1):
            raise ValueError

        i = self.ix
        args = [0] * (self.nargs)
        self.stride *= (self.nargs - 1)

        while i < size - 1:
            args[0] = arr[i]

            for j in range(1, self.nargs):
                args[j] = arr[i + j]

            arr[i + 1] = self.op(*args)
            i += self.stride
        return arr

    @classmethod
    def add(cls):
        return cls(operator.add)

    @classmethod
    def sub(cls):
        return cls(operator.sub)

    @classmethod
    def mul(cls):
        return cls(operator.mul)

    @classmethod
    def div(cls):
        return cls(operator.truediv)

    @classmethod
    def floordiv(cls):
        return cls(operator.floordiv)


def inner_product(arr1, arr2):
    arr_out = pair_wise(arr1, arr2, operator.mul)
    return reductor().add().reduce(arr_out)


'''
Generalised reduction routines:
'''


def reduce_array(arr, faxis, func):
    global j, k, arr_out, new_shape

    make_mdim(arr, arr.mdim + 1)

    if isinstance(faxis, list):
        ndim = len(faxis)
        for i in range(ndim):
            arr = reduce_array(arr, faxis[i], func)
        return arr

    roll_axis(arr, faxis)

    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides
    data = arr.data

    new_shape = list(shape)
    tmp0 = [0] * arr.shape[0]
    axis_counter = [0] * mdim

    def recurse(ix):
        global j, k, arr_out, new_shape

        axis = shape[ix]
        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i

                ix_i = pair_wise_accumulate(axis_counter, strides)

                arr_val = data[ix_i]

                tmp0[j] = arr_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                recurse(ix - 1)

                if ix == 1:
                    j = 0
                    tmp1 = func(tmp0)

                    if k == 0:
                        if isinstance(tmp1, list):
                            new_shape[0] = len(tmp1)
                        elif isinstance(tmp1, md.mdarray):
                            new_shape.pop(0)
                            new_shape = tmp1.shape + new_shape
                        else:
                            new_shape.pop(0)
                        arr_out = zeros(new_shape)

                    if isinstance(tmp1, list):
                        for l in range(len(tmp1)):
                            arr_out.data[k] = tmp1[l]
                            k += 1
                    elif isinstance(tmp1, md.mdarray):
                        for l in range(len(tmp1)):
                            arr_out.data[k] = tmp1.data[l]
                            k += 1
                    else:
                        arr_out.data[k] = tmp1
                        k += 1

    j = k = 0
    recurse(mdim - 1)
    return arr_out


'''
End generalised reduction routines.
'''
