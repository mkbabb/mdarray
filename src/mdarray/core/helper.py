from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

__all__ = [
    "flatten_list",
    "get_strides",
    "make_mdim_shape",
    "pair_wise",
    "remove_extraneous_dims",
    "roll_array",
    "swap",
    "update_dict",
]


def update_dict(d1: dict, d2: dict, recursive: bool = True) -> dict:
    if isinstance(d1, dict) and isinstance(d2, dict):
        for i, j in d1.items():
            if isinstance(j, dict):
                if recursive:
                    d2 = update_dict(j, d2)
                else:
                    d2.update({i: j})
            else:
                with contextlib.suppress(ZeroDivisionError):
                    d2.update({i: j})
    return d2


def get_strides(shape: list[int]) -> list[int]:
    N = len(shape)
    init = 1
    strides = [0] * N
    strides[0] = init

    for i in range(N - 1):
        init *= shape[i]
        strides[i + 1] = init
    return strides


def flatten_shape(shape: list[int], order: int = -1) -> list[int]:
    mdim = len(shape)
    new_mdim = mdim - order

    new_shape = [0] * (mdim - order)
    for i in range(new_mdim):
        new_shape[i] = shape[i]
    prod = 1

    for i in range(new_mdim - 1, mdim):
        prod *= shape[i]
    new_shape[-1] = prod

    return new_shape


def make_mdim_shape(
    shape: list[int],
    ndim: int,
    pad: Any = 1,
) -> list[Any]:
    mdim = len(shape)
    diff = mdim - ndim

    if mdim < ndim:
        shape += [pad] * (ndim - mdim)
    elif mdim > ndim:
        shape = flatten_shape(shape, diff) if isinstance(pad, (int, float)) else shape[:ndim]
    return shape


def swap(seq: list, ix1: int, ix2: int) -> None:
    t = seq[ix1]
    seq[ix1] = seq[ix2]
    seq[ix2] = t


def roll_array(seq: list, axis: int, iterations: int = 1) -> None:
    ndim = len(seq)
    if axis == 0:
        return
    elif axis < 0:
        axis += ndim

    def recurse(ix: int) -> None:
        swap(seq, axis, ix)
        ix += 1
        if ix == axis:
            return
        else:
            recurse(ix)

    for _i in range(iterations):
        recurse(0)


def pair_wise(seq1: list, seq2: list, func: Callable[[Any, Any], Any]) -> list:
    buff = [0] * len(seq1)
    for n, i in enumerate(seq1):
        t = func(i, seq2[n])
        buff[n] = t
    return buff


def remove_extraneous_dims(seq: list) -> list:
    def recurse(seq: list) -> list:
        if len(seq) == 1:
            try:
                seq = recurse(seq[0])
                return seq
            except IndexError:
                return seq
        else:
            return seq

    return recurse(seq)


def flatten_list(seq: list, order: int = 1) -> tuple[list, int, list[int]]:
    shape = [len(seq)]
    mdim = 0

    def recurse(seq: list) -> list:
        nonlocal shape, mdim
        ndim = len(seq)
        buff: list = []
        mdim = 0

        for i in range(ndim):
            seq_i = seq[i]
            if isinstance(seq_i, list):
                buff_r = recurse(seq_i)
                M = len(seq_i)
                if len(shape) <= mdim + 1:
                    shape.insert(-1, M)
                mdim += 1
                buff += [buff_r] if mdim <= order else buff_r
            elif mdim != 0:
                buff += [seq_i]

        if mdim == 0:
            return seq
        else:
            return buff

    flat = recurse(seq)
    return flat, mdim, shape
