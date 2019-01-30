import math
import operator
import random
import sys
from functools import reduce

import numpy as np

from mdarray_core.exceptions import *
from mdarray_core.formatting import *
from mdarray_core.functions import *
from mdarray_core.helper import *
from mdarray_core.indexing import *
from mdarray_core.math import *
from mdarray_core.types import *

__all__ = ["mdarray", "arange", "zeros",
           "full", "ones", "tomdarray", "tondarray"]


class mdarray(object):
    def __init__(self, shape=None, **kwargs):

        self.shape = shape if shape else [1]
        self.size = 1
        self.mdim = 1
        self.strides = [1]
        self.data = [0]
        self.dtype = int

        self.__dict__.update(kwargs)

        if "size" not in kwargs:
            self._get_size()
        if "mdim" not in kwargs:
            self._get_mdim()
        if "strides" not in kwargs:
            self._get_strides()

        self.formatter = None

    def reshape(self, new_shape):
        new_size = reduce(lambda x, y: x*y, new_shape)

        if new_size != self.size:
            raise IncompatibleDimensions

        self.shape = new_shape

        self._get_mdim()
        self._get_strides()
        self.size = new_size
        return self

    def T(self, axis1=0, axis2=1):
        self.strides = swap_item(self.strides, axis1, axis2)
        self.shape = swap_item(self.shape, axis1, axis2)

        return self

    def flatten(self, order=1):
        new_mdim = self.mdim - order
        new_shape = [0]*(self.mdim - order)

        for i in range(new_mdim - 1):
            new_shape[i] = self.shape[i]

        init = 1
        for i in range(order + 1):
            init *= self.shape[self.mdim - (i + 1)]

        new_shape[new_mdim - 1] = init

        return self.reshape(new_shape)

    def to_list(self):
        return make_nested(self)

    def astype(self, type):
        try:
            if type == complex:
                pass
            else:
                pass
        except TypeError:
            raise TypeError
        return self

    def _get_mdim(self):
        self.mdim = len(self.shape)

    def _get_size(self):
        self.size = reduce(lambda x, y: x*y, self.shape)

    def _get_strides(self):
        self.strides = get_strides(self.shape)

    def _get_mdarray_inquery(self):
        return mdarray_inquery(self)

    def __add__(self, other):
        apply_binary_function(self, other, operator.add)
        return self

    def __sub__(self, other):
        apply_binary_function(self, other, operator.sub)
        return self

    def __mul__(self, other):
        apply_binary_function(self, other, operator.mul)
        return self

    def __truediv__(self, other):
        apply_binary_function(self, other, operator.truediv)
        return self

    def __floordiv__(self, other):
        apply_binary_function(self, other, operator.floordiv)
        return self

    def __pow__(self, power, modulo=None):
        apply_binary_function(self, power, operator.pow)
        return self

    def __sin__(self):
        apply_unary_function(self, math.sin)
        return self

    def __cos__(self):
        apply_unary_function(self, math.cos)
        return self

    def __tan__(self):
        apply_unary_function(self, math.tan)
        return self

    def __arcsin__(self):
        apply_unary_function(self, math.asin)
        return self

    def __arccos__(self):
        apply_unary_function(self, math.acos)
        return self

    def __arctan__(self):
        apply_unary_function(self, math.atan)
        return self

    def __sinh__(self):
        apply_unary_function(self, math.sinh)
        return self

    def __cosh__(self):
        apply_unary_function(self, math.cosh)
        return self

    def __tanh__(self):
        apply_unary_function(self, math.tanh)
        return self

    def __arcsinh__(self):
        apply_unary_function(self, math.asinh)
        return self

    def __arccosh__(self):
        apply_unary_function(self, math.acosh)
        return self

    def __arctanh__(self):
        apply_unary_function(self, math.atanh)
        return self

    def __str__(self):
        return print_array(self, ', ', self.formatter)

    # def __setitem__(self, key, value):
    #     tmp = self[key]
    #     print(tmp)

    # def __getitem__(self, item):
    #     if not isinstance(item, gslice):
    #         item = gslice(item, self.a_inqry)
    #     print(item)
    #     data = iter_gslice(arr, item, 1000)
    #     return data

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        ppos = self.pos
        self.pos += 1

        if self.pos == self.size:
            raise StopIteration
        else:
            return self.data[ppos]

    def __len__(self):
        return self.size


def zeros(shape=None, **kwargs):
    mdarr = mdarray(shape=shape, **kwargs)
    mdarr.data = [0]*mdarr.size
    return mdarr


def ones(shape=None, **kwargs):
    mdarr = mdarray(shape=shape, **kwargs)
    mdarr.data = [1]*mdarr.size
    return mdarr


def full(shape=None, fill_value=0, **kwargs):
    mdarr = mdarray(shape=shape, **kwargs)
    mdarr.data = [fill_value]*mdarr.size
    return mdarr


def arange(size):
    data = [i for i in range(size)]
    return mdarray(size=size, data=data)


def tomdarray(arr):
    if isinstance(arr, mdarray):
        return arr
    else:
        arr = list(arr)

        if isinstance(arr, list):
            arr, _, shape = flatten(arr, order=-1)
            md = mdarray(shape=shape, data=arr)
        elif isinstance(arr, dict):
            md = [[i, j] for i, j in arr.items()]

        return tomdarray(md)


def tondarray(arr):
    nd = np.asarray(arr.data).reshape(arr.shape)
    return nd
