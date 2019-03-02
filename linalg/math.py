import random
from functools import partial

import numpy as np

import core
import MultiArray as ma

__all__ = ["dot", "norm"]


def dot(arr1, arr2):
    shape1 = arr1.shape
    shape2 = arr2.shape
    new_shape = [arr2.shape[0], arr1.shape[0]]

    for i in range(1, arr1.mdim):
        if shape1[1] != shape2[1]:
            raise core.IncompatibleDimensions
        elif i > 1:
            new_shape.append(arr1.shape[i])

    arr_out = core.zeros(new_shape, order=arr1.order, dtype=arr1.dtype)
    for i in range(arr2.shape[0]):
        ix = [core.inf] * arr1.mdim
        ix[0] = i
        print(arr2[ix])
        arr_val = arr1 * (arr2[ix].T())
        arr_val = core.reduce_array(arr_val, 0, sum)
        arr_out[ix] = arr_val
    return arr_out


def norm(arr, metric=2):
    arr_out = core.reduce_array(arr**metric, 0, sum)
    return arr_out**(1 / metric)


# mat1 = core.repeat(core.irange([3, 1, 2]), [1], [3])
# mat2 = core.ones([3, 3, 1]) * 2

# import numpy as np


# mat1 = core.tondarray(mat1)
# mat2 = core.tondarray(mat2)

# print(np.dot(mat1, mat2))

# print(dot(mat1, mat2))


def gaussian_elim(arr, rref=True):
    mdim = arr.mdim
    shape = arr.shape
    strides = arr.strides
    col_stride = strides[1]

    row = shape[0]
    col = shape[1]

    data = arr.data

    for i in range(row):
        diag = i * (col_stride + 1)
        pix = i
        pivot = data[diag]

        for j in range(i + 1, col):
            ppix = i + j * col_stride
            ppivot = data[ppix]

            if abs(ppivot) > abs(pivot):
                pix = j
                pivot = ppivot

        if pix != i:
            pix *= col_stride
            for j in range(i, col):
                core.swap_item(data, pix + j, i * col_stride + j)

        if pivot != 0:
            prix = i * col_stride
            for j in range(i + 1, col):
                ppix = j * col_stride
                ppivot = data[ppix + i]

                mul = -ppivot / pivot
                # print(f"ppivot: {ppivot}, pivot: {pivot}")

                data[ppix + i] = 0
                for k in range(i + 1, col):
                    # print(f"rows: {data[ppix + k]}, {data[prix + k]}")
                    data[ppix + k] += mul * data[prix + k]

    if rref:
        data.reverse()

        i = 0
        while i < col:
            diag = i * (col_stride + 1)
            pix = i
            pivot = data[diag]

            if pivot != 0:
                data[diag] = 1
                prix = i * col_stride

                for j in range(i + 1, col):
                    ppix = j * col_stride
                    ppivot = data[ppix + i]
                    mul = -ppivot / pivot
                    data[ppix + i] = 0

                    for k in range(col - (i + 1)):
                        data[ppix + k] += mul * data[prix + k]

            i += 1
        data.reverse()

    return arr


def scramble(arr, axis):
    def scrmble(seq):
        random.shuffle(seq)
        return seq
    return core.reduce_array(arr, axis, scrmble)


def gaussian_elim2(arr):
    rows, cols = arr.shape[0], arr.shape[1]

    for i in range(rows):
        pivot = arr[i, i].data[0]
        for j in range(i + 1, cols):
            ppivot = arr[i, j].data[0]
            if ppivot > pivot:
                pivot = ppivot
                swap_mdim(arr, [..., i], [..., j])

        prow = arr[..., i]
        for j in range(i + 1, cols):
            ppivot = arr[i, j].data[0]
            mul = -ppivot / pivot
            prow_j = arr[..., j]
            arr[..., j] = mul * prow + prow_j
    return arr


def swap_mdim(arr, ix1, ix2):
    t = arr[ix1]
    arr[ix1] = arr[ix2]
    arr[ix2] = t
    return arr


# random.seed(1)
# mat1 = core.irange([3, 3])
# mat1 = scramble(mat1, 1)
# print(mat1)
# sort = core.argsort(mat1[0, core.inf], 1)
# print(sort)
# sort = core.repeat(sort, [0], [3])

# I, J = core.dense_meshgrid(range(3), range(3))

# print(mat1[I, sort])

# mat1 = core.irange([4, 4])
# mat1 = scramble(mat1, 1)
# print(mat1)
# print(mat1)


# def pred(x):
#     if x > -1:
#         return True
# return False


# sm = core.all(mat1, core.inf, pred)
# print(sm)
# mat1 = gaussian_elim(mat1, True)


# mat3 = scramble(core.irange([4, 4]), 1)
# print(mat3)
# w = mat3 > mat1
# wh = core.where(w, 0)
# print(wh)


# print(mat1[wh])
# mat1 = scramble(mat1, 1)
# print(mat1)
# sort = core.argsort(mat1, 0)
# print(sort)
# # sort = core.repeat(sort, [0], [3])

# print(indicies(mat1, sort, 0)
