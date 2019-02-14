from functools import partial, reduce
import random

from core.reduction import reduce_array
from core.types import inf, nan
from core.manipulation import roll_axis
from core.helper import swap_item

__all__ = ["any", "all", "where",
           "argsort", "argmax", "argmin",
           "sort"]


def _pred(x):
    if x > inf:
        return True
    else:
        return False


def _any(pred, lst):
    N = len(lst)
    for i in range(N):
        lst_i = lst[i]
        b_i = pred(lst_i)
        if b_i:
            return True
    return False


def _all(pred, lst):
    N = len(lst)
    for i in range(N):
        lst_i = lst[i]
        b_i = pred(lst_i)
        if not b_i:
            return False
    return True


def any(arr, axis, pred):
    func = partial(_any, pred)
    return reduce_array(arr, axis, func)


def all(arr, axis, pred):
    func = partial(_all, pred)
    return reduce_array(arr, axis, func)


def _where(lst):
    N = len(lst)
    ixs = []
    for i in range(N):
        lst_i = lst[i]
        if lst_i:
            ixs.append(i)
    return ixs


def where(arr, axis=inf):
    return reduce_array(arr, axis, _where)


def argmax(arr, axis):
    def amx(seq):
        N = len(seq)
        _max = 0
        _max_ix = 0
        for i in range(N):
            lst_i = seq[i]
            if lst_i > _max:
                _max = lst_i
                _max_ix = i
        return _max_ix
    return reduce_array(arr, axis, amx)


def argmin(arr, axis):
    def amx(seq):
        N = len(seq)
        _min = inf
        _min_ix = 0
        for i in range(N):
            lst_i = seq[i]
            if lst_i < _min:
                _min = lst_i
                _min_ix = i
        return _min_ix
    return reduce_array(arr, axis, amx)


def argsort(arr, axis, roll=False):
    def asort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)
    arr_out = reduce_array(arr, axis, asort)
    if roll:
        roll_axis(arr_out, axis)
    return arr_out


def scramble(arr, axis):
    def scrmble(seq):
        random.shuffle(seq)
        return seq
    return reduce_array(arr, axis, scrmble)


def partition(seq, key, left, right):
    pix = left
    pivot = key(seq[right], right)

    for i in range(left, right):
        seq_i = key(seq[i], i)
        if seq_i <= pivot:
            swap_item(seq, pix, i)
            pix += 1

    swap_item(seq, pix, right)
    return pix


def quicksort(seq, key, left, right):
    if left < right:
        pix = partition(seq, key, left, right)
        quicksort(seq, key, left, pix - 1)
        quicksort(seq, key, pix + 1, right)


def sort(seq, key, kind="quicksort"):
    size = len(seq)
    if kind == "quicksort":
        quicksort(seq, key, 0, size - 1)
    elif kind == "mergesort":
        pass
