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
    def __init__(self,
                 data: Optional[Union[List[Any], MultiArray]] = None,
                 shape: Optional[List[int]] = None,
                 strides: Optional[List[int]] = None,
                 mdim: Optional[int] = None,
                 size: Optional[int] = None,
                 order: Optional[str] = None):
        if not shape and not size:
            raise TypeError
        elif shape and not size:
            self._shape = shape
            self._size = reduce(lambda x, y: x * y, shape)
        elif size and not shape:
            self._size = size
            self._shape = [size]
        else:
            self._size = size
            self._shape = shape

        self._mdim = len(self._shape)

        if strides:
            self._strides = strides
        else:
            self._strides = core.get_strides(self._shape)

        if not order:
            self._order = "C"
        else:
            self._order = order

        if not data:
            self._data: List[Any] = []
        else:
            self._data = data

        self._stride_shape = core.pair_wise(self._shape,
                                            self._strides,
                                            operator.mul)

        self._axis_counter = [0] * self._mdim
        self._was_advanced = [False] * self._mdim

        self._rept_counter = [0] * self._mdim
        self._repeats = [0] * self._mdim

        self._pos = 0
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
    def data(self) -> list:
        return self._data

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

    def advance(self,
                step: int = 1) -> int:
        i = 0
        while (i < step):
            if (self._rept_counter[0] < self._repeats[0]):
                self._rept_counter[0] += 1
            else:
                self._rept_counter[0] = 0
                self._axis_counter[0] += 1

            for j in range(1, self.mdim):
                if self._axis_counter[j - 1] >= self._stride_shape[j - 1]:
                    if (self._rept_counter[j] == self._repeats[j]):
                        self._rept_counter[j] = 0
                        self._axis_counter[j - 1] = 0
                        self._axis_counter[j] += self.strides[j]

                        self._was_advanced[j] = True
                    else:
                        self._rept_counter[j] += 1
                        self._axis_counter[j - 1] = 0

                        self._was_advanced[j] = True
                else:
                    self._was_advanced[j] = False
            i += 1

        self._index = sum(self._axis_counter)
        self._pos += step

        return self._index

    def at(self, pos: Union[list, int]) -> MultiArray:
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

    def __next__(self) -> MultiArray:
        if self._index < self._size:
            self.advance(1)
            return self
        else:
            raise StopIteration

    def __iter__(self):
        yield self._data[self._index]
        self.__next__()

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

    def reshape(self, new_shape: List[int]) -> MultiArray:
        core.reshape(self, new_shape)
        return self

    def T(self, axis1: int = 0, axis2: int = 1) -> MultiArray:
        core.transpose(self, axis1, axis2)
        return self

    def flatten(self, order: int = -1) -> MultiArray:
        core.flatten(self, order)
        return self

    def to_list(self) -> List[Any]:
        return core.make_nested_list(self)

    '''
    Operator overloads
    '''

    # def __repr__(self) -> str:
    #     s = "MultiArray("
    #     s += core.print_array(self)
    #     s += ")"
    #     return s

    # def __str__(self) -> str:
    #     s = core.print_array(self)
    #     return s
