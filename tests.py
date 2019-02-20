from functools import reduce

from core import *
from linalg import *
from mdarray import *
import numpy as np
import random


'''
Concatenation tests:
'''
# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = irange(size).reshape(shape)

# c1 = full(shape=[1, 3, 2], fill_value=99)
# c2 = full(shape=[2, 3, 2], fill_value=33)
# concat = concatenate(c1, c2, caxis=0)
# print(concat)


'''
End concatenation tests.
'''


'''
Repeat tests:
'''
# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = irange(size).reshape(shape)

# repeated = repeat(arr, raxis=1, rept=2)
# print(repeated)

# np_arr = np.irange(0, size).reshape(shape[::-1])
# np_repeated = np.repeat(np_arr, 2, 1)
# print(np_repeated)
'''
End repeat tests.
'''


'''
Meshgrid tests:
'''
# iter1 = range(3)
# iter2 = range(4)
# iter3 = range(2)


# grid = meshgrid(iter1, iter2, iter3)
# for i in grid:
#     print(i)
# # print(grid[0])

# iter1 = range(3)
# iter2 = range(4)
# iter3 = range(2)

# np_grid = np.meshgrid(iter2, iter3, iter1)
# print(np_grid)
'''
End meshgrid tests.
'''


'''
Pad array tests:
'''
# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = irange(size).reshape(shape)
# np_arr = tondarray(arr)

# pad1 = (1, 1)
# pad2 = (2, 1)
# pad3 = (1, 2)

# # v1 = full(shape=[1, 3, 2], fill_value=99)
# # v2 = full(shape=[7, 1, 2], fill_value=99)
# # v3 = full(shape=[7, 5, 1], fill_value=99)

# # v7 = concatenate(v1, arr, v1, caxis=0)
# # print(v7)
# # v8 = concatenate(v2, v7, v2, caxis=1)
# # print(v8)
# # v9 = concatenate(v3, v8, v3, caxis=2)
# # print(v9)

# padded = pad_array(arr, (pad1, pad2, pad3), 99)
# print(padded)

# print('\n\n')

# np_padded = np.pad(np_arr, (pad1, pad2, pad3),
#                    'constant', constant_values=(99,))
# print(np_padded)
# print(np_padded.shape)
'''
End pad array tests.
'''

'''
Reduce array tests:
'''
# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = irange(size).reshape(shape)
# np_arr = tondarray(arr)

# print(arr)

# faxis = 2
# reduced = reduce_array(arr, faxis, sum, mode="value")
# print(reduced)


'''
To mdarray and ndarray tests:
'''

# lst = [[[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]],

#        [[10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]]]

# v = [[[1,  2,  3],
#       [4,  5,  6],
#       [7,  8,  9]],

#      [[10, 11, 12],
#       [13, 14, 15],
#       [16, 17, 18]]]

'''
End to mdarray and ndarray tests.
'''

'''
Broadcasting tests:
'''

# shape1 = [6, 4, 1, 2, 1, 2]
# size1 = reduce(lambda x, y: x*y, shape1)
# X = irange(size1).reshape(shape1)

# shape1 = [1, 4, 3, 2, 2, 1]
# size1 = reduce(lambda x, y: x*y, shape1)
# Y = irange(size1).reshape(shape1)


# np_X = md.tondarray(X)
# np_Y = md.tondarray(Y)


# def func(x, y): return x*y


# v = broadcast(X, Y, func)
# print(v.shape)

# print(v)
# print('---\n')


# np_v = (np_X * np_Y)

# print(np_v)
# print(np_v.shape)


# More tests

# a0 = irange(24).reshape([2, 3, 4, 1])
# a1 = ones([2, 3, 1])*12
# a2 = ones([1, 3, 1])*12
# a3 = zeros([2, 3, 4])
# a4 = linear_range(0, 10, 20).reshape([1, 1, 4, 5])


# a0, a4 = broadcast_arrays(a0, a4)
# print(a0)

# arr1 = irange(100).reshape([2, 25, 2])
# arr2 = ones([1, 25, 1])*1.0

# arr1 = md.tondarray(arr1)
# arr2 = md.tondarray(arr2)
# print(arr1*arr2)

# print(arr1*arr2)

'''
End broadcasting tests.
'''

'''
Reduction tests:
'''

# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = irange(size).reshape(shape)
# np_arr = md.tondarray(arr)

# arr = md.tomdarray([[1,2,3], [4,5,6], [7,8,9]])


# func = lambda x: irange(9).reshape([3, 3])
# rarr = reduce_array(arr, 0, func)
# print(np_arr.sum(-1))
# print(rarr)
# diag = diagonal([0, 1, 2, 3])
# print(diag)

# arr = md.tondarray(arr)
# print(np.diag(arr))

# arr = irange(24).reshape([4, 3, 2])
# np_arr = tondarray(arr)
# print(arr)

# v = reduce_array(arr, 2, reductor.add().accumulate)
# print(v)
# print(np_arr.sum(-1))


# arr = tomdarray([10]).reshape([1, 1])
# arr = repeat(arr, [0, 1], [4, 4])
# print(arr)

# print(10*identity(5))

# mshape = [5, 4, 3, 2, 2, 5]

# arr1 = irange([5, 1, 3, 2, 2, 1])
# np_arr1 = tondarray(arr1)
# arr2 = ones([1, 4, 3, 2, 1, 5])
# arr3 = ones([5, 4, 1, 1, 2, 5])
# arr4 = ones([5, 4, 3, 2, 1, 1])

# new_shape, repts = generate_broadcast_shape(arr1, arr2, arr3, arr4)

# func = lambda x, y: x+y
# v = broadcast_bnry(arr1, arr2, arr3, arr4, func=func)
# print(v)
# print(np.broadcast_to(np_arr1, mshape[::-1]))

'''
End reduction tests:
'''


'''
Begin slicing tests:
'''

# mat = [[0, 1, 2],
#        [3, 4, 5],
#        [5, 6, 7]]


# v = [[1],
#      [2],
#      [3]]


# mat = tomdarray(mat)
# v = tomdarray(v)
# v = repeat(v, [0], [3])
# print(v)


# for i in range(mat.shape[0]):
#     print(mat[i, inf])


# arr = irange([4, 3]) + 1000
# rows = inf
# cols = 2


# print(arr[rows, cols])
# arr[rows, cols] = 99
# print(arr)
# print(arr)
# ixs = [rows, cols, depths, depths2]
# slc = slice_array(ixs, arr)
# print(slc)

# arr = irange([3, 4])
# rows = tomdarray([[0, 2],
#                   [0, 2]])

# cols = tomdarray([[0, 0],
#                   [3, 3]])

# rows = [0, 2]
# cols = [0, 3]
# print(arr[rows, cols])

# arr[rows, cols] = 999
# print(arr, "lol")

'''
End slicing tests.
'''


# arr1 = irange([2, 5, 2])
# arr2 = irange([2, 5, 2]) * 99

# arr1 = irange([5, 2, 2])
# arr2 = irange([5, 2, 1])
# print(arr1)
# print("\n")
# print(arr2)
# concat = concatenate(arr1, arr2, arr1, caxis=2)
# print("\n")
# print(concat)

'''
'''

# pad = [[1, 2], [1, 1], [1, 1]]
# arr = irange([3, 3, 2])

# padded = pad_array(arr, pad, pad_wrap)
# print(padded)


# np_padarr = np.pad(np_arr, ((1, 1), (1, 1), (1, 1)), mode="reflect")
# print(np_padarr)

# print(arr)
# out, slc = pad_array2(arr, pad)
# print(out)


'''
'''

'''
Outer product tests:
'''
# random.seed(1)


# def scramble(arr, axis):
#     def scrmble(seq):
#         random.shuffle(seq)
#         return seq
#     return reduce_array(arr, axis, scrmble)


# arr1 = irange([5, 1])
# arr2 = irange([1, 5])

# arr1 = scramble(arr1, 0)
# arr2 = scramble(arr2, 1)

# narr1 = tondarray(arr1)
# narr2 = tondarray(arr2)

# print(arr1)
# print(arr2)


# out = arr2.T()*arr1.T()
# print(out)

# nout = np.kron(narr1, narr2)
# print(nout)
'''
End outer product tests.
'''

'''
Sort by tests
'''


def complete_ix(arr, ixs, axis=-1):
    mdim = arr.mdim
    axis1 = arr.shape[axis]
    axis2 = ixs.shape[axis]

    ixs_part = linear_range(axis1 - axis2, axis1)
    make_mdim(ixs_part, mdim)
    swap_axis(ixs_part, 0, axis)

    new_shape = list(arr.shape)
    new_shape[0] = arr.shape[0] - ixs.shape[0]
    ixs_part = broadcast_toshape(ixs_part, new_shape)

    return concatenate(ixs, ixs_part, caxis=0)


random.seed(2)

# arr = irange([5, 3])
# arr = scramble(arr, 1)
# print(arr)


# def rfunc(seq):
#     seq1 = diagonal(tomdarray(seq))
#     print(seq1.shape)
#     return seq1


# rarr = reduce_array(arr, 1, rfunc)
# narr = tondarray(arr)
# rnarr = np.apply_along_axis(np.diag, 0, narr)
# print(rnarr)
# print(rnarr.shape)

# col1 = arr[0, ...]
# col2 = arr[1, ...]


# print("\n")

# srt = lexical_sort(col1, col2, axis=0)
# print(srt)


# srt = complete_ix(arr, srt, 1)
# print(srt)

# ixs = indicies(arr, srt, 1)
# print(ixs)
# print(ixs)


# srt = sort2(arr[1, ..., ...], axis=1, roll=False)
# print(srt)
# part = tomdarray([2, 3]).reshape([1, 1, 2])
# part = repeat(part, [0, 1], [4, 4])

# srt = concatenate(srt, part, caxis=0)
# print(srt)


# ixs = indicies(arr, srt, axis=1)
# print(ixs)

arr = tomdarray([[4, 2, 3],
                 [4, 2, 5],
                 [3, 5, 5],
                 [1, 5, 5],
                 [3, 2, 1],
                 [5, 2, 2],
                 [3, 2, 3],
                 [4, 3, 4],
                 [3, 4, 1],
                 [5, 3, 4]])

# x, y, z = arr[0, ...], arr[1, ...], arr[2, ...]
# srt = lexical_sort(x, y, z, axis=0)
# print(srt)
# ixs = indicies(arr, srt, 0)
# print(ixs)


arr = tomdarray([[4, 2, 5],
                 [4, 2, 3],
                 [3, 5, 5],
                 [3, 5, 4]])


def lexical_sort(*keys, axis):
    keys = tuple(keys)
    arr = concatenate(*keys, caxis=0)
    print(arr)

    mdim = arr.mdim
    shape = keys[0].shape

    tmp = irange(shape[axis])
    make_mdim(tmp, mdim)
    swap_axis(tmp, 0, axis)
    ix_arr = broadcast_toshape(tmp, shape)
    tix1 = [...] * mdim

    def key(seq, ix):
        tix1[axis] = ix
        return seq[tix1]

    quicksort(arr, ix_arr, key, axis, 0, shape[axis] - 1)
    return ix_arr


print(arr)
x, y = arr[0, ...], arr[2, ...]
ix_arr = lexical_sort(x, y, axis=1)
print(ix_arr)
ixs = indicies(arr, ix_arr, 1)
print(ixs)

# def mykey(seq, i):
#     return seq[[0, 2], i]

# axis = 1

# quicksort(arr, mykey, axis, 0, arr.shape[axis] - 1)
# print(arr)


# srt = lexical_sort(arr[0, ...], axis=1)
# print(srt)
# ixs = indicies(arr[2, ...], srt, 1)
# print(ixs)
# srt2 = lexical_sort(ixs, axis=1)
# print(srt2)
# ixs = indicies(arr, srt, 1)
# print(ixs)
