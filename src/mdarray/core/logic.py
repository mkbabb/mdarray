"""Sorting, searching, and logical operations on mdarray.

Provides quicksort with Lomuto partitioning, argmax/argmin, where
(boolean indexing), and universal any/all predicates.
"""

from __future__ import annotations

import random
from functools import partial
from typing import Any

from ..array import mdarray
from .helper import swap

__all__ = [
    "argmax",
    "argmin",
    "argsort",
    "mdall",
    "mdany",
    "quicksort",
    "scramble",
    "sort",
    "where",
]


def _any(pred: Any, lst: list) -> bool:
    N = len(lst)
    for i in range(N):
        lst_i = lst[i]
        b_i = pred(lst_i)
        if b_i:
            return True
    return False


def _all(pred: Any, lst: list) -> bool:
    N = len(lst)
    for i in range(N):
        lst_i = lst[i]
        b_i = pred(lst_i)
        if not b_i:
            return False
    return True


def mdany(arr: mdarray, axis: int = 0, pred: Any = None) -> Any:
    if not pred:

        def pred(x: Any) -> Any:
            return x

    func = partial(_any, pred)
    return func(arr.data)


def mdall(arr: mdarray, axis: int = 0, pred: Any = None) -> Any:
    if not pred:

        def pred(x: Any) -> Any:
            return x

    func = partial(_all, pred)
    return func(arr.data)


def where(arr: mdarray, axis: Any = None) -> list[int]:
    N = len(arr.data)
    ixs = []
    for i in range(N):
        if arr.data[i]:
            ixs.append(i)
    return ixs


def argmax(arr: mdarray, axis: int = 0) -> int:
    N = len(arr.data)
    _max = arr.data[0]
    _max_ix = 0
    for i in range(1, N):
        if arr.data[i] > _max:
            _max = arr.data[i]
            _max_ix = i
    return _max_ix


def argmin(arr: mdarray, axis: int = 0) -> int:
    N = len(arr.data)
    _min = arr.data[0]
    _min_ix = 0
    for i in range(1, N):
        if arr.data[i] < _min:
            _min = arr.data[i]
            _min_ix = i
    return _min_ix


def argsort(arr: mdarray, axis: int = 0, roll: bool = False) -> list[int]:
    return sorted(range(len(arr.data)), key=arr.data.__getitem__)


def scramble(arr: mdarray, axis: int = 0) -> mdarray:
    data = list(arr.data)
    random.shuffle(data)
    arr._data = data
    return arr


def partition(seq: list, ixs: list, key: Any, axis: int, left: int, right: int) -> int:
    """Lomuto partition: rearrange *seq* so elements <= pivot precede those > pivot.

    Returns the final pivot index.  Both *seq* and *ixs* (index tracker)
    are permuted in tandem.
    """
    pix = left
    pivot = key(seq, right)

    for i in range(left, right):
        seq_i = key(seq, i)
        eq = seq_i <= pivot
        if not isinstance(eq, bool):
            eq = all(eq)
        if eq:
            swap(seq, pix, i)
            swap(ixs, pix, i)
            pix += 1

    swap(seq, pix, right)
    swap(ixs, pix, right)
    return pix


def quicksort(seq: list, ixs: list, key: Any, axis: int, left: int, right: int) -> None:
    """Recursive quicksort using Lomuto partitioning.

    Sorts *seq* in-place and permutes *ixs* in tandem for index tracking.
    """
    if left < right:
        pix = partition(seq, ixs, key, axis, left, right)
        quicksort(seq, ixs, key, axis, left, pix - 1)
        quicksort(seq, ixs, key, axis, pix + 1, right)


def sort(seq: list, ixs: list, key: Any, axis: int, kind: str = "quicksort") -> None:
    size = len(seq)
    if kind == "quicksort":
        quicksort(seq, ixs, key, axis, 0, size - 1)
