from __future__ import annotations

import math
import operator
import random
import sys
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import core


class MultiArray(object):
    def __init__(self, shape: Optional[List[int]] = None,
                 size: Optional[int] = None,
                 data: Optional[List[Any]] = None,
                 dtype: Any = None,
                 order: Optional[str] = None):

        if not shape and not size:
            raise TypeError
        elif shape and not size:
            self._shape = shape
            self._size = reduce(lambda x, y: x * y, shape)
        elif size and not shape:
            self._size = size
            self._shape = [size]

        if not order:
            self._order = "C"
        else:
            self._order = order

        if not data:
            self._data: List[Any] = []
        else:
            self._data = data

        if not dtype:
            self._dtype = int
        else:
            self._dtype = dtype

        self._mdim = len(self._shape)
        self._strides = core.get_strides(self._shape)
        self.formatter = None
        self._iterator = MultiArrayIter(self._shape,
                                        self._strides,
                                        self._mdim,
                                        self._size,
                                        self._order)

    @property
    def iterator(self):
        return self._iterator

    @iterator.setter
    def iterator(self, other):
        self._iterator._update(other)

    @property
    def shape(self) -> List[int]:
        return self._shape

    @shape.setter
    def shape(self, other: List[int]) -> None:
        core.reshape(self, other)
        self.iterator = self

    @property
    def size(self) -> int:
        return self._size

    @property
    def mdim(self) -> int:
        return self._mdim

    @mdim.setter
    def mdim(self, other: int) -> None:
        diff = self._mdim - other
        if diff < 0:
            self.shape = self._shape + [1] * abs(diff)
        else:
            self.flatten(diff)
        self.iterator = self

    @property
    def strides(self) -> List[int]:
        return self._strides

    @strides.setter
    def strides(self, other: List[int]) -> None:
        if len(other) != self._mdim:
            raise core.IncompatibleDimensions
        else:
            self._strides = other
            self.iterator = self

    @property
    def data(self) -> List[Any]:
        return self._data

    @data.setter
    def data(self, other: List[Any]) -> None:
        other, mdim, shape = core.flatten_list(other, -1)
        if mdim > 1:
            if mdim > self._mdim:
                raise core.IncompatibleDimensions()
        elif shape[0] != self._size:
            raise core.IncompatibleDimensions(
                "This data is too large/small for this array")
        else:
            self._data = other

    @property
    def dtype(self) -> Any:
        return self._dtype

    @dtype.setter
    def dtype(self, other: Any) -> None:
        self.astype(other)

    @property
    def order(self) -> str:
        return self._order

    @order.setter
    def order(self, other: str) -> None:
        if other != self._order:
            if other == "F":
                if self.mdim == 1:
                    self.reshape(self._shape + [1])
                core.swap_item(self._shape, 0, 1)
                self.reshape(self._shape)
            elif self._order == "C" or self.order == "NP":
                self.reshape(self._shape[::-1])
        self._order = other
        self.iterator = self

    def reshape(self, new_shape: List[int]) -> MultiArray:
        core.reshape(self, new_shape)
        self.iterator = self
        return self

    def T(self, axis1: int = 0, axis2: int = 1) -> MultiArray:
        core.transpose(self, axis1, axis2)
        self.iterator = self
        return self

    def flatten(self, order: int = -1) -> MultiArray:
        core.flatten(self, order)
        self.iterator = self
        return self

    def to_list(self) -> List[Any]:
        return core.make_nested_list(self)

    def astype(self, dtype) -> MultiArray:
        return core.astype(self, dtype)

    '''
    Operator overloads
    '''

    def __repr__(self) -> str:
        return core.print_array(self, ', ', self.formatter)

    def __str__(self) -> str:
        return self.__repr__()

    def __setitem__(self, slc, arr2) -> MultiArray:
        core.slice_array(slc, self, arr2, True)
        return self

    def __getitem__(self, slc) -> MultiArray:
        return core.slice_array(slc, self, None, False)

    def __iter__(self) -> MultiArray:
        self.pos = 0
        return self

    def __next__(self) -> List[Any]:
        ppos = self.pos
        self.pos += 1

        if self.pos == self.size + 1:
            raise StopIteration
        else:
            return self.data[ppos]

    def __len__(self) -> int:
        return self._size

    '''
    Logical operators
    '''

    def __eq__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.eq)

    def __le__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.le)

    def __ge__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.ge)

    def __lt__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.lt)

    def __gt__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.gt)

    '''
    Mathematical operators
    '''

    def __add__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.add)

    def __iadd__(self, other: Any) -> MultiArray:
        return self.__add__(other)

    def __radd__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(other, self, operator.add)

    def __sub__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.sub)

    def __rsub__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(other, self, operator.sub)

    def __mul__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.mul)

    def __rmul__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(other, self, operator.mul)

    def __truediv__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.truediv)

    def __rtruediv__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(other, self, operator.truediv)

    def __floordiv__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(other, self, operator.floordiv)

    def __pow__(self, power: Union[float, int]) -> MultiArray:
        return core.apply_binary_function(self, power, operator.pow)

    def __rpow__(self, power: Union[float, int]) -> MultiArray:
        return core.apply_binary_function(power, self, operator.pow)

    def __sqrt__(self) -> MultiArray:
        return core.apply_unary_function(self, math.sqrt)

    def __nroot__(self, root: Union[float, int]) -> MultiArray:
        return self.__pow__(1 / root)

    def __sin__(self) -> MultiArray:
        core.apply_unary_function(self, math.sin)
        return self

    def __cos__(self) -> MultiArray:
        core.apply_unary_function(self, math.cos)
        return self

    def __tan__(self) -> MultiArray:
        core.apply_unary_function(self, math.tan)
        return self

    def __arcsin__(self) -> MultiArray:
        core.apply_unary_function(self, math.asin)
        return self

    def __arccos__(self) -> MultiArray:
        core.apply_unary_function(self, math.acos)
        return self

    def __arctan__(self) -> MultiArray:
        core.apply_unary_function(self, math.atan)
        return self

    def __arctan2__(self, other: Any) -> MultiArray:
        return core.apply_binary_function(self, other, math.atan2)

    def __sinh__(self) -> MultiArray:
        core.apply_unary_function(self, math.sinh)
        return self

    def __cosh__(self) -> MultiArray:
        core.apply_unary_function(self, math.cosh)
        return self

    def __tanh__(self) -> MultiArray:
        core.apply_unary_function(self, math.tanh)
        return self

    def __arcsinh__(self) -> MultiArray:
        core.apply_unary_function(self, math.asinh)
        return self

    def __arccosh__(self) -> MultiArray:
        core.apply_unary_function(self, math.acosh)
        return self

    def __arctanh__(self) -> MultiArray:
        core.apply_unary_function(self, math.atanh)
        return self


class MultiArrayIter(object):
    def __init__(self, shape: Optional[List[int]],
                 strides: List[int],
                 mdim: int,
                 size: Optional[int],
                 order: Optional[str] = None):
        self._shape = shape
        self._strides = strides
        self._mdim = mdim
        self._size = size

        self._stride_shape = core.pair_wise(self._shape,
                                            self._strides,
                                            operator.mul)
        self._axis_counter = [0] * self._mdim
        self._was_advanced = [False] * self._mdim

        self._rept_counter = [0] * self._mdim
        self._repeats = [1] * self._mdim
        self._repeat = False

        self._pos = 0
        self._rpos = 0
        self._index = 0

    @property
    def shape(self) -> list:
        return self._shape

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def mdim(self) -> int:
        return self._mdim

    @property
    def size(self) -> int:
        return self._size

    @property
    def repeats(self) -> list:
        return self._repeats

    @repeats.setter
    def repeats(self, other: list) -> None:
        self._repeat = True
        other = core.make_mdim_shape(other, self._mdim)
        self._repeats = other

    @property
    def axis_counter(self) -> list:
        return self._axis_counter

    @property
    def was_advanced(self) -> list:
        return self._was_advanced

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def index(self) -> int:
        return self._index

    def _update(self, arr) -> MultiArrayIter:
        if arr.mdim != self._mdim:
            self._mdim = arr.mdim
            self._axis_counter = core.make_mdim_shape(
                self._axis_counter, self._mdim, 0)
            self._was_advanced = core.make_mdim_shape(
                self._was_advanced, self._mdim, False)
            self._rept_counter = core.make_mdim_shape(
                self._rept_counter, self._mdim, 0)
            self._repeats = core.make_mdim_shape(
                self._repeats, self._mdim, 1)
        self._shape = arr.shape
        self._strides = arr.strides
        self._order = arr.order
        self._stride_shape = core.pair_wise(self._shape,
                                            self._strides,
                                            operator.mul)
        if arr.size != self._size:
            self._size = arr.size
            self.at(0)
        return self

    def advance(self, step: int = 1) -> int:
        i = 0
        while i < step:
            self._axis_counter[0] += self._strides[0]
            for j in range(1, self._mdim):
                if self._axis_counter[j - 1] >= self._stride_shape[j - 1]:
                    self._axis_counter[j - 1] = 0
                    self._axis_counter[j] += self._strides[j]
                    self._was_advanced[j] = True
                else:
                    self._was_advanced[j] = False
            # axis-repeat routine:
            if self._repeat:
                if self._rept_counter[0] == self._repeats[0] - 1:
                    self._rept_counter[0] = 0
                    self._rpos += 1
                else:
                    self._rept_counter[0] += 1

                for j in range(1, self._mdim):
                    stride_j = self._strides[j]
                    if self._was_advanced[j] and self.zero_axes_before(j):
                        self._rpos -= stride_j
                        if self._rept_counter[j] == self._repeats[j] - 1:
                            self._rept_counter[j] = 0
                            self._rpos += self._strides[j]
                        else:
                            self._rept_counter[j] += 1
            i += 1
        if self._repeat:
            self.at(self._rpos)
        else:
            self._index = sum(self._axis_counter)
        self._pos += step
        return self._index

    def at(self, pos: Union[list, int]) -> MultiArrayIter:
        if isinstance(pos, list):
            self._pos = core.unravel([pos], self.shape)[0]
        else:
            self._pos = pos

        if self._pos == 0:
            for i in range(self._mdim):
                self._axis_counter[i] = 0
                self._was_advanced[i] = False
            self._index = 0
            self._pos = 0
        else:
            core.ravel_internal(self._pos, self._axis_counter,
                                self._mdim, self._strides)
            for i in range(1, self._mdim):
                self.was_advanced_before(i)

            self._index = sum(self._axis_counter)
            self._pos -= 1
        return self

    def __next__(self) -> MultiArrayIter:
        if self._index < self._size:
            self.advance(1)
            return self
        else:
            raise StopIteration

    def __iter__(self):
        while self._index < self._size:
            yield self
            self.__next__()

    def __repr__(self) -> str:
        s = f"unraveled index: {self._index}"
        return s

    def zero_axes_before(self, axis: int) -> bool:
        for i in range(axis):
            if i != axis:
                if self._rept_counter[i] != 0:
                    return False
        return True

    def was_advanced_before(self, axis: int) -> bool:
        for i in range(axis):
            if self._axis_counter[i] != (self._stride_shape[i] - 1):
                self._was_advanced[axis] = False
                return False
        self._was_advanced[axis] = True
        return True
