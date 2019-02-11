import math
import operator
import random
import sys
from functools import reduce

import numpy as np
import core


__all__ = ["mdarray"]


class mdarray(object):
    def __init__(self, shape=None, **kwargs):
        self.shape = shape
        self.size = 1
        self.mdim = 1
        self.strides = [1]
        self.data = [0]
        self.dtype = int
        self._order = "C"

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
        
        self.order = self._order
        self.formatter = None

    def reshape(self, new_shape):
        core.reshape(self, new_shape)
        return self

    def T(self, axis1=0, axis2=1):
        core.transpose(self, axis1, axis2)
        return self

    def flatten(self, order=-1):
        core.flatten(self, order)
        return self

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, other):
        if other != self.order:
            if self.mdim == 1:
                self.reshape(self.shape + [1])
            self.T()
        self._order = other

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

    '''
    Operator overloads
    '''

    def __repr__(self):
        return core.print_array(self, ', ', self.formatter)

    def __str__(self):
        return self.__repr__()

    def __setitem__(self, slc, arr2):
        core.slice_array(slc, self, arr2, True)
        return self

    def __getitem__(self, slc):
        return core.slice_array(slc, self, None, False)

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

    '''
    Logical operators
    '''

    def __eq__(self, other):
        return core.apply_binary_function(self, other, operator.eq)

    def __le__(self, other):
        return core.apply_binary_function(self, other, operator.le)

    def __ge__(self, other):
        return core.apply_binary_function(self, other, operator.ge)

    def __lt__(self, other):
        return core.apply_binary_function(self, other, operator.lt)

    def __gt__(self, other):
        return core.apply_binary_function(self, other, operator.gt)

    '''
    Mathematical operators
    '''

    def __add__(self, other):
        return core.apply_binary_function(self, other, operator.add)

    def __iadd__(self, other):
        return self.__add__(other)

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
