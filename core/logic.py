import random
from functools import partial, reduce

import mdarray as md
from core.helper import swap_item
from core.manipulation import roll_axis
from core.reduction import reduce_array
from core.types import inf, nan

__all__ = ["mdany", "mdall", "where",
           "argsort", "argmax", "argmin",
           "sort", "scramble", "quicksort"]


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


def mdany(arr, axis, pred):
    func = partial(_any, pred)
    return reduce_array(arr, axis, func)


def cont_any(arr):
    for i in arr:
        if not i:
            return False
    return True


def mdall(arr, axis, pred):
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
    return reduce_array(arr, axis, scrmble, True)


def mdswap_item(arr, axis, ix1, ix2):
    if ix1 == ix2 or not arr:
        return
    else:
        if isinstance(arr, md.mdarray):
            if axis != nan:
                mdim = arr.mdim
                tix1 = [...] * mdim
                tix2 = [...] * mdim
                tix1[axis] = ix1
                tix2[axis] = ix2
                ix1 = tix1
                ix2 = tix2
        swap_item(arr, ix1, ix2)


def partition(seq, ixs, key, axis, left, right):
    pix = left
    pivot = key(seq, right)

    for i in range(left, right):
        seq_i = key(seq, i)
        eq = (seq_i <= pivot)
        if not isinstance(eq, bool):
            eq = any(eq)
        if eq:
            mdswap_item(seq, axis, pix, i)
            mdswap_item(ixs, axis, pix, i)
            pix += 1

    mdswap_item(seq, axis, pix, right)
    mdswap_item(ixs, axis, pix, right)
    return pix


def quicksort(seq, ixs, key, axis, left, right):
    if left < right:
        pix = partition(seq, ixs, key, axis, left, right)
        quicksort(seq, ixs, key, axis, left, pix - 1)
        quicksort(seq, ixs, key, axis, pix + 1, right)


def sort(seq, ixs, key, axis, kind="quicksort"):
    size = len(seq)
    if kind == "quicksort":
        quicksort(seq, ixs, key, 0, lambda x: x, size - 1)
    elif kind == "mergesort":
        pass
