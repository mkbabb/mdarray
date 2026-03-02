from __future__ import annotations

import operator
from inspect import signature
from typing import Any

from .helper import pair_wise

__all__ = ["inner_product", "reductor"]


class reductor:
    def __init__(
        self,
        op: Any = None,
        ix: int = 0,
        init: Any = 0,
        stride: int = 1,
        exclude: list[int] | None = None,
    ) -> None:
        self.op = op
        self.ix = ix
        self.init = init
        self.stride = stride
        self.exclude = exclude

        if self.exclude is None:
            self.exclude = [-1]

        if self.op:
            self.nargs = len(signature(self.op).parameters)

    def reduce(self, arr: list) -> Any:
        size = len(arr)
        t = (size // self.stride) % self.nargs

        if (t != 0) & (t != self.nargs - 1):
            raise ValueError

        i = self.ix
        out = self.init
        args = [self.init] * (self.nargs)
        self.stride *= self.nargs - 1

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

    def accumulate(self, arr: list) -> list:
        size = len(arr)
        t = (size // self.stride) % self.nargs

        if (t != 0) & (t != self.nargs - 1):
            raise ValueError

        i = self.ix
        args = [self.init] * (self.nargs)
        self.stride *= self.nargs - 1

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
    def add(
        cls, ix: int = 0, init: int = 0, stride: int = 1, exclude: list[int] | None = None
    ) -> reductor:
        return cls(operator.add, ix, init, stride, exclude)

    @classmethod
    def sub(
        cls, ix: int = 0, init: int = 0, stride: int = 1, exclude: list[int] | None = None
    ) -> reductor:
        return cls(operator.sub, ix, init, stride, exclude)

    @classmethod
    def mul(
        cls, ix: int = 0, init: int = 1, stride: int = 1, exclude: list[int] | None = None
    ) -> reductor:
        return cls(operator.mul, ix, init, stride, exclude)

    @classmethod
    def div(
        cls, ix: int = 0, init: int = 1, stride: int = 1, exclude: list[int] | None = None
    ) -> reductor:
        return cls(operator.truediv, ix, init, stride, exclude)

    @classmethod
    def floordiv(
        cls, ix: int = 0, init: int = 1, stride: int = 1, exclude: list[int] | None = None
    ) -> reductor:
        return cls(operator.floordiv, ix, init, stride, exclude)


def inner_product(arr1: list, arr2: list) -> Any:
    arr_out = pair_wise(arr1, arr2, operator.mul)
    return reductor.add().reduce(arr_out)
