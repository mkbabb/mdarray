from functools import reduce

from core import *
from mdarray import *
import numpy as np


'''
Concatenation tests:
'''
# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = arange(size).reshape(shape)

# c1 = full(shape=[1, 3, 2], fill_value=99)
# c2 = full(shape=[2, 3, 2], fill_value=99)
# print(c1)
'''
End concatenation tests.
'''


'''
Repeat tests:
'''
# shape = [5, 3, 2]
# size = reduce(lambda x, y: x*y, shape)
# arr = arange(size).reshape(shape)

# repeated = repeat(arr, raxis=1, rept=2)
# print(repeated)

# np_arr = np.arange(0, size).reshape(shape[::-1])
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
# arr = arange(size).reshape(shape)
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
# arr = arange(size).reshape(shape)
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
# X = arange(size1).reshape(shape1)

# shape1 = [1, 4, 3, 2, 2, 1]
# size1 = reduce(lambda x, y: x*y, shape1)
# Y = arange(size1).reshape(shape1)


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

# a0 = arange(24).reshape([2, 3, 4, 1])
# a1 = ones([2, 3, 1])*12
# a2 = ones([1, 3, 1])*12
# a3 = zeros([2, 3, 4])
# a4 = linear_range(0, 10, 20).reshape([1, 1, 4, 5])


# a0, a4 = broadcast_arrays(a0, a4)
# print(a0)

# arr1 = arange(100).reshape([2, 25, 2])
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
# arr = arange(size).reshape(shape)
# np_arr = md.tondarray(arr)

# arr = md.tomdarray([[1,2,3], [4,5,6], [7,8,9]])


# func = lambda x: arange(9).reshape([3, 3])
# rarr = reduce_array(arr, 0, func)
# print(np_arr.sum(-1))
# print(rarr)
# diag = diagonal([0, 1, 2, 3])
# print(diag)

# arr = md.tondarray(arr)
# print(np.diag(arr))

# arr = arange(24).reshape([4, 3, 2])
# np_arr = tondarray(arr)
# print(arr)

# v = reduce_array(arr, 2, reductor.add().accumulate)
# print(v)
# print(np_arr.sum(-1))


# arr = tomdarray([10]).reshape([1, 1])
# arr = repeat(arr, [0, 1], [4, 4])
# print(arr)

# print(10*eye(5))

# mshape = [5, 4, 3, 2, 2, 5]

# arr1 = arange([5, 1, 3, 2, 2, 1])
# np_arr1 = tondarray(arr1)
# arr2 = ones([1, 4, 3, 2, 1, 5])
# arr3 = ones([5, 4, 1, 1, 2, 5])
# arr4 = ones([5, 4, 3, 2, 1, 1])

# new_shape, repts = generate_broadcast_shape(arr1, arr2, arr3, arr4)

# func = lambda x, y: x+y
# v = broadcast_bnry(arr1, arr2, arr3, arr4, func=func)
# print(v)
# print(np.broadcast_to(np_arr1, mshape[::-1]))

mat = [[0, 1, 2],
       [3, 4, 5],
       [5, 6, 7]]


v = [[1],
     [2],
     [3]]


mat = tomdarray(mat)
v = tomdarray(v)


arr = arange([4, 3]) + 1000
rows = inf
cols = 2


print(arr[rows, cols])
arr[rows, cols] = 99
print(arr)
# print(arr)
# ixs = [rows, cols, depths, depths2]
# slc = slice_array(ixs, arr)
# print(slc)

# arr = arange([3, 4, 2])
# rows = tomdarray([[0, 2],
#                   [0, 2]])

# cols = tomdarray([[0, 0],
#                   [3, 3]])


# ixs = [[0, 2], inf, inf]

# ixs = [rows, cols]
# slc = slice_array(ixs, arr)
# print(slc)


# arr = tondarray(arr)
# rows = tondarray(rows)
# cols = tondarray(cols)

# print(arr[1, cols, rows])

# print(arr[rows, cols])

# ixs = expand_dims(ixs, arr)
# # roll_array(ixs, 1)


# ixs = dense_meshgrid(*ixs)

# for i in ixs:
#     print(i)
# ixs = broadcast_arrays(*ixs)


# arr_out = ix_(*ixs)
# print(arr_out)
# print(np.ix_([0, 1, 2], [0, 1, 0]))


# print(arr.strides)
# ix = [ix1]
# ix = expand_dims(ix, arr)
# # print(ix)
# slc = expand_slice_array2(ix, 3, arr.strides)
# print(slc)

# print(nparr[ix1, ix2])


'''
End reduction tests:
'''
