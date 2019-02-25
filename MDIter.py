from __future__ import annotations
from core import *
import mdarray as md
import typing
from functools import reduce


class MDIter(object):
    def __init__(self, arr: md.mdarray):
        self._arr = arr
        self._shape = arr.shape
        self._size = arr.size
        self._mdim = arr.mdim

        self._axis_counter = [0] * self._arr.mdim
        self._was_advanced = [False] * self._arr.mdim

        for n, i in enumerate(self._arr.strides):
            if i == 1:
                self._was_advanced[n] = True

        self._rept_counter = [0] * self._arr.mdim
        self._repeats = [1] * self._arr.mdim
        self._repeat = False

        self._pos = 0
        self._rpos = 0
        self._index = 0

    @property
    def arr(self) -> md.mdarray:
        return self._arr

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    @property
    def mdim(self) -> int:
        return self._mdim

    @property
    def repeats(self) -> list:
        return self._repeats

    @repeats.setter
    def repeats(self, other: list) -> None:
        self._repeat = True
        other = make_mdim_shape(other, self._arr.mdim)
        self._shape = pair_wise(other, self._shape, operator.mul)
        self._size = reduce(lambda x, y: x * y, self._shape)
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

    def advance(self, step: int = 1) -> int:
        repeated = False
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
            if self._repeat:
                if self._rept_counter[0] == self._repeats[0] - 1:
                    self._rept_counter[0] = 0
                    self._rpos += 1
                else:
                    self._rept_counter[0] += 1

                for j in range(1, self._arr.mdim):
                    stride_j = self._arr.strides[j]
                    if self._was_advanced[j] and self.zero_axes_before(j):
                        self._rpos -= stride_j
                        if self._rept_counter[j] == self._repeats[j] - 1:
                            self._rept_counter[j] = 0
                            self._rpos += self._arr.strides[j]
                        else:
                            self._rept_counter[j] += 1
            i += 1
        if self._repeat:
            self.at(self._rpos)
        else:
            self._index = inner_product(self._axis_counter,
                                        self._arr.strides)
        self._pos += i
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

        ravel_internal(self._pos, self._axis_counter,
                       self.arr.mdim, self.arr.strides)
        for i in range(1, self._arr.mdim):
            self.was_advanced_before(i)

        self._index = inner_product(self._axis_counter,
                                    self._arr.strides)
        self._pos -= 1
        return self

    def __next__(self) -> MDIter:
        if self._pos < self.arr.size:
            self.advance(1)
            return self
        else:
            raise StopIteration

    def __iter__(self) -> typing.Generator:
        for i in range(self._size):
            yield self
            next(self)

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
            if self._axis_counter[i] != (self._arr.shape[i] - 1):
                self._was_advanced[axis] = False
                return False
        self._was_advanced[axis] = True
        return True


def broadcast_iter(arrs):
    mditers = [MDIter(i) for i in arrs]
    new_shape, repts = generate_broadcast_shape(*arrs)
    for n, i in enumerate(repts):
        mditers[n].repeats = i
    return mditers


def broadcast_nary(arrs, func):
    mditers = broadcast_iter(arrs)
    shape, size = mditers[0].shape, mditers[0].size
    arr_out = zeros(shape)
    ndim = len(mditers)
    fargs = [0] * ndim

    for n, i in enumerate(zip(*mditers)):
        for m, j in enumerate(i):
            fargs[m] = arrs[m].data[j.index]
        arr_out.data[n] = func(fargs)
    return arr_out


def broadcast_toshape_iter(arr, shape):
    mditer = MDIter(arr)
    arr_shape = md.mdarray(shape=shape, order=arr.order, dtype=arr.dtype)
    new_shape, repts = generate_broadcast_shape(arr, arr_shape)
    mditer.repeats = repts[0]
    return mditer


def concatenate_iter(*arrs, caxis):
    arrs = tuple(arrs)
    ndim = len(arrs)
    arr1 = arrs[0]
    mdim = arr1.mdim
    if caxis < 0:
        caxis += mdim
    new_shape = list(arr1.shape)
    new_shape[caxis] = 0
    for i in range(ndim):
        arr_i = arrs[i]
        if mdim != arr_i.mdim:
            raise IncompatibleDimensions(
                "The dimensions of array one do not equal the dimensions of array two!")
        for j in range(mdim):
            if j != caxis:
                if new_shape[j] != arr_i.shape[j]:
                    raise IncompatibleDimensions(
                        "All axes but caxis must be equivalent to concatenate the arrays.")
        new_shape[caxis] += arr_i.shape[caxis]

    arr_out = zeros(shape=new_shape, order=arr1.order, dtype=arr1.dtype)

    if caxis != mdim - 1:
        iters = [MDIter(i) for i in arrs]
        k = 0
        while k < arr_out.size:
            for i in range(ndim):
                for j in iters[i]:
                    if j.was_advanced[caxis + 1]:
                        j._was_advanced[caxis + 1] = False
                        break
                    else:
                        arr_out.data[k] = arrs[i].data[j.index]
                        k += 1
    else:
        k = 0
        while k < arr_out.size:
            for i in range(ndim):
                for j in arrs[i].data:
                    arr_out.data[k] = j
                    k += 1
    return arr_out


arr1 = irange([2, 2, 3, 1])
arr2 = irange([2, 1, 1, 3])
arr3 = irange([2, 2, 3, 3])
# caxis = 1


# base_arr1 = concatenate(arr1, arr2, arr3, caxis=caxis)
# test_arr1 = concatenate_iter(arr1, arr2, arr3, caxis=caxis)
# eq = (base_arr1.data == test_arr1.data)
# print(eq)

broadcast_nary([arr1, arr2, arr3], func=sum)

b1, b2, b3 = broadcast_arrays(arr1, arr2, arr3)
for i in zip(b1.data, b2.data, b3.data):
    print(i)
