import operator
import random
from functools import reduce

import numpy as np

from core import *
from linalg import *
from mdarray import *


arr = tomdarray([[4, 2, 3],
                 [4, 2, 5],
                 [3, 5, 5],
                 [1, 5, 5],
                 [3, 2, 1],
                 [5, 2, 2],
                 [3, 2, 3],
                 [4, 3, 4],
                 [3, 4, 1],
                 [5, 3, 4]])

# arr2 = [0, 1], ...
# slc, new_shape, oned = expand_indicies(arr2, arr)
# slc = dense_meshgrid(*slc)
# arr_out = zeros(shape=new_shape)
# unravel_dense(*slc, arr_in=arr, arr_out=arr_out, set=False)


class MDIter(object):
    def __init__(self, mdarray=None):
        self._mdarray = arr
        self._axis_counter = [0] * self._mdarray.mdim
        self._was_advanced = [False] * self._mdarray.mdim
        self._pos = 0
        self._index = 0

    @property
    def mdarray(self):
        return self._mdarray

    @property
    def axis_counter(self):
        return self._axis_counter

    @property
    def was_advanced(self):
        return self._was_advanced

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, other):
        if other > self._mdarray.size:
            raise ValueError
        else:
            ravel_internal(other, self._axis_counter,
                           self.mdarray.mdim, self.mdarray.strides)
            self._index = inner_product(self._axis_counter,
                                        self._mdarray.strides)
        self.__first = True if other == 0 else self.__first
        self._pos = other

    @property
    def index(self):
        return self._index

    def grapple(self, buff, axis, func, count=1):
        if not func:
            func = lambda x: x
        if axis < 0:
            axis += self.mdarray.mdim

        k = 0
        flag_count = 0
        for i in self:
            flag = True
            buff[k] = self.mdarray.data[self._index]
            for j in range(axis):
                if self._axis_counter[j] != 0:
                    flag = False
            if flag:
                flag_count += 1
            if flag and flag_count == count:
                fbuff = func(buff)
                return fbuff
            k += 1

    def advance(self, step=1):
        i = 0
        while i < step:
            self._axis_counter[0] += 1
            self._was_advanced[0] = True
            for j in range(self._mdarray.mdim - 1):
                if self._axis_counter[j] >= self._mdarray.shape[j]:
                    self._axis_counter[j] = 0
                    self._was_advanced[j + 1] = True
                    self._axis_counter[j + 1] += 1
                else:
                    self._was_advanced[j + 1] = False

            self._index = inner_product(self._axis_counter,
                                        self._mdarray.strides)
            self._pos += 1
            i += 1
        return self.index

    def __next__(self):
        if self._pos < self.mdarray.size + 1:
            self.advance(1)
            return self
        else:
            raise StopIteration

    def __repr__(self):
        s = f"unraveled index: {self._index}"
        return s


def get_ret_shaped2(buff, arr, new_shape, axis, keepdims):
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


def _insert_into_flattened2(buff, arr_out, j):
    if isinstance(buff, md.mdarray):
        buff = buff.data

    if isinstance(buff, list):
        for n, i in enumerate(buff):
            arr_out.data[n + j] = i
        j += len(buff)
    else:
        arr_out.data[j] = buff
        j += 1
    return j


def reduce_iter(arr, faxis, func, keepdims=False):
    mdim = arr.mdim
    roll_axis(arr, faxis)
    mditer = MDIter(arr)
    shape = arr.shape
    new_shape = list(shape)
    buff = [0] * shape[0]

    j = 0
    for i in range(shape[faxis]):
        fbuff = mditer.grapple(buff, 1, func)
        if i == 0:
            arr_out = get_ret_shaped2(fbuff, arr, new_shape,
                                      faxis, keepdims)
        j = _insert_into_flattened2(fbuff, arr_out, j)

    roll_axis(arr, faxis, mdim - 1)
    return arr_out


def repeat_iter(arr, raxes, repts):
    ndim = len(raxes)
    strides = arr.strides
    mditer = MDIter(arr)

    start = 0
    for i in range(arr.size + 1):
        for j in range(ndim):
            raxis = raxes[j]
            rept = repts[j]

            if mditer.was_advanced[raxis]:
                end = mditer.pos
                for k in range(rept):
                    mditer.pos = start
                    for l in range(strides[raxis]):
                        print(mditer.index)
                        next(mditer)
                mditer.pos = end
                start = end
        next(mditer)


arr = irange([5, 5])
print(arr)
narr = tondarray(arr)
mditer = MDIter(arr)
# for i in mditer:
#     print(i.pos, i.index, i.axis_counter, i.was_advanced)

# print(np.repeat(narr, 2, 1))


repeat_iter(arr, [0], [2])
# t = reduce_iter(arr, 0, sum, True)
# print(t.shape)
# print(t)
