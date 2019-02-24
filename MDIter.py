from __future__ import annotations
from core import *
import mdarray as md
import typing
from functools import reduce


class MDIter(object):
    def __init__(self, arr: md.mdarray):
        self._arr: md.mdarray = arr
        self._axis_counter: list = [0] * self._arr.mdim
        self._was_advanced: list = [False] * self._arr.mdim
        self._was_advanced[0] = True

        self._pos: int = 0
        self._index: int = 0
        self._size: int = (self._arr.size - 1)

        self._rept_counter = [0] * self._arr.mdim
        self._repeats = [1] * self._arr.mdim
        self._rpos = 0

    @property
    def arr(self) -> md.mdarray:
        return self._arr

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

    @property
    def size(self) -> int:
        return self._size

    @property
    def repeats(self) -> list:
        return self._repeats

    @repeats.setter
    def repeats(self, other: list) -> None:
        rsize = reduce(lambda x, y: x * y, other)
        self._repeats = other
        self._size = (self._size + 1)*rsize

    def advance(self, step: int = 1) -> int:
        i = 0
        while i < step:
            self._axis_counter[0] += 1
            for j in range(self._arr.mdim - 1):
                if self._axis_counter[j] >= self._arr.shape[j]:
                    self._axis_counter[j] = 0
                    self._axis_counter[j + 1] += 1
                    self._was_advanced[j + 1] = True
                else:
                    self._was_advanced[j + 1] = False
        # axis-repeat routine:
            if self._rept_counter[0] == self._repeats[0] - 1:
                self._rept_counter[0] = 0
                self._rpos += 1
            else:
                self._rept_counter[0] += 1
                self.at(self._rpos)

            for j in range(1, self._arr.mdim):
                stride_j = self._arr.strides[j]
                if self.was_advanced[j] and zero_axes_before(self._rept_counter, j):
                    self._rpos -= stride_j
                    if self._rept_counter[j] == self._repeats[j] - 1:
                        self._rept_counter[j] = 0
                        self._rpos += self._arr.strides[j]
                    else:
                        self._rept_counter[j] += 1
                        self.at(self._rpos)
            i += 1
        self._index = inner_product(self._axis_counter,
                                    self._arr.strides)
        self._pos += i
        self._size -= i
        return self.index

    def grapple(self, buff: list, axis: int,
                func: typing.Callable[[list], list] = None,
                count: int = 1) -> list:
        if not func:
            func = lambda x: x
        if axis < 0:
            axis += self.arr.mdim
        k = 0
        _count = 0
        for i in self:
            if self.was_advanced[axis]:
                _count += 1
                if _count == count:
                    fbuff = func(buff)
                    return fbuff
            buff[k] = self.arr.data[self._index]
            k += 1
        return buff

    def at(self, pos: typing.Union[list, int]) -> MDIter:
        if isinstance(pos, list):
            self._pos = unravel([pos], self.arr.shape)[0]
        else:
            self._pos = pos

        self._size = (self.arr.size - 1) - self._pos
        ravel_internal(self._pos, self._axis_counter,
                       self.arr.mdim, self.arr.strides)

        for i in range(1, self._arr.mdim):
            if self._axis_counter[i] == (self._arr.shape[i] - 1):
                self._was_advanced[i] = True
            else:
                self._was_advanced[i] = False

        self._index = inner_product(self._axis_counter,
                                    self._arr.strides)
        self._pos -= 1

        return self

    def __next__(self) -> MDIter:
        if self._pos < self.arr.size:
            self.advance(1)
            if self._size == 0:
                self._was_advanced[-1] = True
            return self
        else:
            raise StopIteration

    def __iter__(self):
        for i in range(self.size):
            yield self
            next(self)

    def __repr__(self) -> str:
        s = f"unraveled index: {self._index}"
        return s


def zero_axes_before(axis_counter, axis):
    for i in range(axis):
        if i != axis:
            if axis_counter[i] != 0:
                return False
    return True


repts = [1, 3, 10]
arr = irange([10, 5, 3])
rarr = repeat(arr, [0, 1, 2], repts)

# print(arr)
# print(arr.shape)

mditer = MDIter(arr)
mditer.repeats = repts
print(mditer.size, rarr.size)
arr_out = zeros(list(rarr.shape))

for n, i in enumerate(mditer):
    # print(i.index)
    arr_out.data[n] = i.index
    pass
# print(rarr)


for i in range(rarr.size):
    print(rarr.data[i] == arr_out.data[i])
