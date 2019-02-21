import operator
from inspect import signature

import numpy as np

import mdarray as md
from core.creation import make_mdim, tomdarray, zeros
from core.helper import pair_wise
from core.manipulation import roll_axis, swap_axis
from core.types import inf, nan

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
    def __init__(self, op=None, ix=0, init=0, stride=1, exclude=None):
        self.op = op
        self.ix = ix
        self.init = init
        self.stride = stride
        self.exclude = exclude

        if self.exclude == None:
            self.exclude = [-1]

        if self.op:
            self.nargs = len(signature(self.op).parameters)

    def reduce(self, arr):
        size = len(arr)
        t = (size // self.stride) % self.nargs

        if (t != 0) & (t != self.nargs - 1):
            raise ValueError

        i = self.ix
        out = self.init
        args = [self.init] * (self.nargs)
        self.stride *= (self.nargs - 1)

        while i < size - 1:
            for j in range(self.nargs):
                ij = i + j
                exclude = False
                args[j] = self.init
                for k in self.exclude:
                    if ij == k:
                        exclude = True
                        break
                if j == 0:
                    args[0] = arr[0] if i == 0 and not exclude else out
                elif not exclude:
                    args[j] = arr[ij]
            out = self.op(*args)
            i += self.stride
        return out

    def accumulate(self, arr):
        size = len(arr)
        t = (size // self.stride) % self.nargs

        if (t != 0) & (t != self.nargs - 1):
            raise ValueError

        i = self.ix
        args = [self.init] * (self.nargs)
        self.stride *= (self.nargs - 1)

        while i < size - 1:
            for j in range(self.nargs):
                ij = i + j
                exclude = False
                args[j] = self.init
                for k in self.exclude:
                    if ij == k:
                        exclude = True
                        break
                if not exclude:
                    args[j] = arr[ij]
                elif j == 0:
                    args[0] = arr[j]

            arr[i + 1] = self.op(*args)
            i += self.stride
        return arr

    @classmethod
    def add(cls, ix=0, init=0, stride=1, exclude=None):
        return cls(operator.add, ix, init, stride, exclude)

    @classmethod
    def sub(cls, ix=0, init=0, stride=1, exclude=None):
        return cls(operator.sub, ix, init, stride, exclude)

    @classmethod
    def mul(cls, ix=0, init=1, stride=1, exclude=None):
        return cls(operator.mul, ix, init, stride, exclude)

    @classmethod
    def div(cls, ix=0, init=1, stride=1, exclude=None):
        return cls(operator.truediv, ix, init, stride, exclude)

    @classmethod
    def floordiv(cls, ix=0, init=1, stride=1, exclude=None):
        return cls(operator.floordiv, ix, init, stride, exclude)


def inner_product(arr1, arr2):
    arr_out = pair_wise(arr1, arr2, operator.mul)
    return reductor().add().reduce(arr_out)


'''
Generalised reduction routines:
'''


'''
Helper functions for reduce_array:
'''


def get_ret_shaped(buff, arr, new_shape, axis, keepdims):
    buff = tomdarray(buff)
    if buff.size > 1:
        if not keepdims:
            new_shape.pop(axis)
            new_shape = buff.shape + new_shape
    else:
        if keepdims:
            new_shape[axis] = 1
        else:
            new_shape.pop(axis)
    arr_out = zeros(new_shape)
    return arr_out


def _insert_into_flattened(buff, arr_out, ixs, j, keepdims):
    if isinstance(buff, md.mdarray):
        buff = buff.data

    if isinstance(buff, list):
        if keepdims and len(buff) == len(ixs):
            for n, i in enumerate(ixs):
                arr_out.data[i] = buff[n]
        else:
            for n, i in enumerate(buff):
                arr_out.data[n + j] = i
        j += len(buff)
    else:
        arr_out.data[j] = buff
        j += 1
    return j


'''
'''


def reduce_array(arr, faxis, func, keepdims=False):
    global j, arr_out

    mdim = arr.mdim

    if faxis == inf:
        arr_out = func(arr.data)
        return arr_out
    elif faxis < 0:
        faxis += mdim

    new_shape = list(arr.shape)
    roll_axis(arr, faxis)

    shape = arr.shape
    strides = arr.strides
    data = arr.data

    buff = [0] * shape[0]
    ixs = [0] * shape[0]
    axis_counter = [0] * mdim

    def recurse(ix):
        global j, arr_out
        axis = shape[ix]

        if ix == 0:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)

                buff[i] = data[ix_i]
                ixs[i] = ix_i

            fbuff = func(buff)

            if j == 0:
                arr_out = get_ret_shaped(fbuff, arr, new_shape, faxis, keepdims)
            j = _insert_into_flattened(fbuff, arr_out, ixs, j, keepdims)

        else:
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]
                recurse(ix - 1)

    j = 0
    recurse(mdim - 1)
    roll_axis(arr, faxis, mdim - 1)

    return arr_out


'''
End generalised reduction routines.
'''
