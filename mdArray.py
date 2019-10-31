from __future__ import annotations

import math
import operator
import random
import sys
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import core


class multiArrayIter(object):
    def __init__(self,
                 shape: List[int],
                 strides: List[int],
                 mdim: int,
                 size: int,
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

    def _update(self, arr) -> multiArrayIter:
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

    def at(self, pos: Union[list, int]) -> multiArrayIter:
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

    def __next__(self) -> multiArrayIter:
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


class multiArray(multiArrayIter):
    pass


def print_array2(maiter,
                 data,
                 sep: Optional[str] = ", ",
                 formatter=None
                 ) -> str:
    if not formatter:
        formatter = lambda x, y: f"{x[y]}"
    if not sep:
        sep = ", "

    mdim = maiter.mdim
    size = maiter.size

    s = ""
    strings = [""] * (mdim - 1)
    for i in range(size):
        s += formatter(data, maiter.index)
        next(maiter)
        s += sep if not maiter.was_advanced[1] else ''

        ix = 0
        for j in range(1, mdim):
            if maiter.was_advanced[j]:
                if j == 1:
                    strings[0] += f"[{s}]"
                    s = ""
                else:
                    strings[j - 1] += f"[{strings[j - 2]}]"
                    strings[j - 2] = ""
                ix += 1
        if ix > 0 and i != size - 1:
            new_line = "\n" * ix
            hanging_indent = " " * (mdim - ix)
            strings[ix - 1] += (sep.strip() + new_line + hanging_indent)

    s = f"[{strings[-1]}]"
    maiter.at(0)
    return s


vi = {0: 1, 1: 2, 4: 3, 5: 4, 10: 5, 11: 6, 14: 7, 15: 8}

values = [1, 2, 3, 4, 5, 6, 7, 8]
ixs = [0, 1, 4, 5, 10, 11, 14, 15]
shape = [2, 2, 2]
mdim = len(shape)
strides = core.get_strides(shape)
size = core.reductor.mul().reduce(shape)

maiter = multiArrayIter(shape, strides, mdim, size)

for i in maiter:
    print(i)


# def fmtter(data, index):
#     v = data.get(index)
#     if v:
#         return str(v)
#     else:
#         return "0"


# s = print_array2(maiter, vi, ", ", fmtter)
# print(s)
