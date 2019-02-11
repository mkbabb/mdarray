import operator
from inspect import signature

import numpy as np

import mdarray as md
from core.creation import make_mdim, zeros, tomdarray
from core.helper import pair_wise, pair_wise_accumulate
from core.manipulation import roll_axis
from core.types import nan, inf

__all__ = ["reductor", "inner_product",
           "reduce_array"]


'''
Generalised reduction of a 1-d array.
The reductor class can either apply a reduction or an accumulation
across a 1-d array. Functions of an airty >= 2 are accepted for both routines.

Parameters:
    op: function=None
        n-ary function that must return a 0-d output.
    
    ix: integer=0
        Starting index of the reductor.
    
    stride: integer=1
        Stride of the reductor.
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
        out = arr[i]
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
    def add(cls, ix=0, stride=1):
        return cls(operator.add, ix, stride)

    @classmethod
    def sub(cls, ix=0, stride=1):
        return cls(operator.sub, ix, stride)

    @classmethod
    def mul(cls, ix=0, stride=1):
        return cls(operator.mul, ix, stride)

    @classmethod
    def div(cls, ix=0, stride=1):
        return cls(operator.truediv, ix, stride)

    @classmethod
    def floordiv(cls, ix=0, stride=1):
        return cls(operator.floordiv, ix, stride)


def inner_product(arr1, arr2):
    arr_out = pair_wise(arr1, arr2, operator.mul)
    return reductor().add().reduce(arr_out)


'''
Generalised reduction routines:
'''


'''
Helper functions for reduce_array:
'''


def get_ret_shaped(arr, axis, shape):
    if isinstance(arr, list):
        shape[axis] = len(arr)
    elif isinstance(arr, md.mdarray):
        shape.pop(axis)
        shape = arr.shape + shape
        shape.insert(axis, 1)
    else:
        shape.pop(axis)
        shape.insert(axis, 1)
    arr_out = zeros(shape)
    return arr_out


def insert_into_flattened(arr_in, arr_out, ix):
    if isinstance(arr_in, list):
        for l in range(len(arr_in)):
            arr_out.data[ix] = arr_in[l]
            ix += 1
    elif isinstance(arr_in, md.mdarray):
        for l in range(len(arr_in)):
            arr_out.data[ix] = arr_in.data[l]
            ix += 1
    else:
        arr_out.data[ix] = arr_in
        ix += 1
    return ix


'''
'''


def reduce_array(arr, faxis, func):
    global j, k, arr_out, new_shape

    if isinstance(faxis, list):
        ndim = len(faxis)
        for i in range(ndim):
            arr = reduce_array(arr, faxis[i], func)
        return arr

    mdim = arr.mdim

    if faxis == inf:
        arr_out = func(arr.data)
        return arr_out

    if faxis < 0:
        faxis += mdim

    new_shape = list(arr.shape)
    roll_axis(arr, faxis)

    shape = arr.shape
    strides = arr.strides
    data = arr.data

    tmp0 = [0] * shape[0]
    axis_counter = [0] * mdim

    def recurse(ix):
        global j, k, arr_out, new_shape
        axis = shape[ix]

        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)
                arr_val = data[ix_i]

                tmp0[j] = arr_val
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                recurse(ix - 1)

                if ix == 1:
                    j = 0
                    tmp1 = func(tmp0)
                    if k == 0:
                        arr_out = get_ret_shaped(tmp1, faxis, new_shape)
                    k = insert_into_flattened(tmp1, arr_out, k)

    j = k = 0
    recurse(mdim - 1)
    roll_axis(arr, faxis, mdim - 1)
    return arr_out


'''
End generalised reduction routines.
'''
