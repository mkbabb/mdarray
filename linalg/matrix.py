import random
from functools import partial

import numpy as np

import core
import multiArray as ma


__all__ = ["diagonal", "identity"]


def diagonal(arr):
    mdim = arr.mdim
    size = arr.size
    shape = arr.shape

    if mdim == 1:
        arr_out = core.zeros([size, size], order=arr.order)
        col_stride = arr_out.strides[1]
        for i in range(size):
            arr_out.data[i * (col_stride + 1)] = arr.data[i]
    elif mdim == 2:
        arr_out = core.zeros([shape[0]], order=arr.order)
        col_stride = arr.strides[1]
        for i in range(shape[0]):
            arr_out.data[i] = arr.data[i * (col_stride + 1)]

    return arr_out


def identity(order=2):
    return diagonal(core.ones([order]))
