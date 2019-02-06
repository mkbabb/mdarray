import math
import operator
import random
import sys
from functools import reduce

import numpy as np
import core


__all__ = ["mdarray", "tomdarray", "tondarray"]


class mdarray(object):
    def __init__(self, shape=None, **kwargs):
        self.shape = shape
        self.size = 1
        self.mdim = 1
        self.strides = [1]
        self.data = [0]
        self.dtype = int
        self.order = "C"

        self.__dict__.update(kwargs)

        if "size" not in kwargs:
            self._get_size()

        if self.shape == None:
            self.shape = [self.size]

        if "mdim" not in kwargs:
            self._get_mdim()
        if "strides" not in kwargs:
            self._get_strides()
        if "dtype" not in kwargs:
            self._get_dtype()

        self.set_order(self.order)
        self.formatter = None

    def reshape(self, new_shape):
        core.reshape(self, new_shape)
        return self

    def T(self, axis1=0, axis2=1):
        core.transpose(self, axis1, axis2)
        return self

    def flatten(self, order=1):
        core.flatten(self, order)
        return self

    def set_order(self, order="C"):
        if order == "F":
            if self.mdim == 1:
                self.reshape(self.shape + [1])
            self.T()
        return self

    def to_list(self):
        return core.make_nested_list(self)

    def astype(self, dtype):
        return core.astype(self, dtype)

    def _get_mdim(self):
        self.mdim = len(self.shape)

    def _get_size(self):
        self.size = reduce(lambda x, y: x * y, self.shape)

    def _get_strides(self):
        self.strides = core.get_strides(self.shape)

    def _get_dtype(self):
        self.dtype = type(self.data[0])

    def __repr__(self):
        return core.print_array(self, ', ', self.formatter)

    def __str__(self):
        return self.__repr__()

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

    def __add__(self, other):
        return core.apply_binary_function(self, other, operator.add)

    def __radd__(self, other):
        return core.apply_binary_function(other, self, operator.add)

    def __sub__(self, other):
        return core.apply_binary_function(self, other, operator.sub)

    def __rsub__(self, other):
        return core.apply_binary_function(other, self, operator.sub)

    def __mul__(self, other):
        return core.apply_binary_function(self, other, operator.mul)

    def __rmul__(self, other):
        return core.apply_binary_function(other, self, operator.mul)

    def __truediv__(self, other):
        return core.apply_binary_function(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return core.apply_binary_function(other, self, operator.truediv)

    def __floordiv__(self, other):
        return core.apply_binary_function(self, other, operator.floordiv)

    def __rfloordiv__(self, other):
        return core.apply_binary_function(other, self, operator.floordiv)

    def __pow__(self, power):
        return core.apply_binary_function(self, power, operator.pow)

    def __rpow__(self, power):
        return core.apply_binary_function(power, self, operator.pow)

    def __sqrt__(self):
        return core.apply_unary_function(self, math.sqrt)

    def __nroot__(self, root):
        return self.__pow__(1 / root)

    def __sin__(self):
        core.apply_unary_function(self, math.sin)
        return self

    def __cos__(self):
        core.apply_unary_function(self, math.cos)
        return self

    def __tan__(self):
        core.apply_unary_function(self, math.tan)
        return self

    def __arcsin__(self):
        core.apply_unary_function(self, math.asin)
        return self

    def __arccos__(self):
        core.apply_unary_function(self, math.acos)
        return self

    def __arctan__(self):
        core.apply_unary_function(self, math.atan)
        return self

    def __arctan2__(self, other):
        return core.apply_binary_function(self, other, math.atan2)

    def __sinh__(self):
        core.apply_unary_function(self, math.sinh)
        return self

    def __cosh__(self):
        core.apply_unary_function(self, math.cosh)
        return self

    def __tanh__(self):
        core.apply_unary_function(self, math.tanh)
        return self

    def __arcsinh__(self):
        core.apply_unary_function(self, math.asinh)
        return self

    def __arccosh__(self):
        core.apply_unary_function(self, math.acosh)
        return self

    def __arctanh__(self):
        core.apply_unary_function(self, math.atanh)
        return self


def tomdarray(arr):
    if isinstance(arr, mdarray):
        return arr
    else:
        if isinstance(arr, list) or isinstance(arr, tuple):
            arr, mdim, shape = core.flatten_list(arr, order=-1)
            arr_out = mdarray(shape=shape, data=arr)
        elif isinstance(arr, dict):
            arr_out = [[i, j] for i, j in arr.items()]
        else:
            arr_out = mdarray(shape=[1], data=[arr])

        return tomdarray(arr_out)


def tondarray(arr):
    nd = np.asarray(arr.data).reshape(arr.shape[::-1])
    return nd
