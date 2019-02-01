from functools import reduce

from mdarray_core import *
from mdarray import *


def func(arr):
    ndim = len(arr)
    mx = 0
    axis = 0
    for i in range(ndim):
        arr_i = arr[i]
        if arr_i > mx:
            mx = arr_i
            axis = i
    return axis


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

'''
End broadcasting tests.
'''

# shape1 = [3]
# size1 = reduce(lambda x, y: x*y, shape1)
# X = arange(size1).reshape(shape1)

# p = broadcast_toshape(X, [3, 3])
# print(p)





a0 = zeros([1, 6, 1])
a1 = zeros([5, 6, 7])
a2 = zeros([1, 6, 1])
a3 = zeros([5, 1, 7])
a4 = zeros([5, 6, 1])

new_shape, raxes, repts = generate_broadcast_shape2(a0, a1, a2, a3, a4)
print(new_shape)
print(raxes)
print(repts)
