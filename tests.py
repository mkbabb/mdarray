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

# shape = [20, 2, 3]

# size = reduce(lambda x, y: x*y, shape)

# arr = arange(size).reshape(shape)
# arr.flatten = pad_array_fmt(arr)
# np_arr = tondarray(arr)


# print(t1)
# print(arr)

# shape = [3, 5]
# size = reduce(lambda x, y: x*y, shape)
# arr = arange(size).reshape(shape)
# print(arr)


# t = tile(arr, (2, 1, 2))
# print(t)


# shape = [6, 4]

# size = reduce(lambda x, y: x*y, shape)

# arr = arange(size).reshape(shape)
# np_arr = md.tondarray(arr)

# ixs = np.where(np_arr > 2)
# t1 = expand_slice_array([*ixs], 2)
# print(md.tomdarray(t1))

# _arr = [[1, 2], [3, 4],
#         [9, 8], [7, 6]]
# arr = md.tomdarray(_arr)
# print(arr)

# slc = [[True, False], [True, True]]
# slc = md.tomdarray(slc)
# v = repeat(slc, 1, 2)
# print(v)


# def pred(x): return True if x > 2 else False


# v = mask(arr, pred)
# print(v)

def repeat_nocopy(arr, raxes, repts):
    global j
    mdim = arr.mdim
    ndim = len(raxes)
    axis_counter = [0]*mdim

    def recurse(ix):
        global j

        shape = arr.shape
        strides = arr.strides
        data = arr.data
        axis = shape[ix]

        remaining_axes = mdim - ix

        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix3 = pair_wise_accumulate(axis_counter, strides)

                try:
                    a_val = data[ix3]
                except:
                    a_val = nan
                print(a_val, j)
                j += 1
        else:
            for i in range(axis):
                axis_counter[ix] = i
                repeated = False
                for k in range(ndim):
                    raxis = raxes[k]
                    rept = repts[k]
                    if ix == raxis or (raxis == 0 and ix != mdim - 1):
                        for l in range(rept):
                            recurse(ix - 1)
                        repeated = True
                        break

                if not repeated:
                    recurse(ix - 1)
    j = 0
    recurse(mdim - 1)


def broadcast_copy(arr1, arr2):
    mdim1 = arr1.mdim
    mdim2 = arr2.mdim
    shape1 = arr1.shape
    shape2 = arr2.shape

    mdim = mdim1

    if mdim1 > mdim2:
        arr2.reshape(shape2 + [1]*(mdim1 - mdim2))
    elif mdim1 < mdim2:
        arr1.reshape(shape1 + [1]*(mdim2 - mdim1))
        mdim = mdim2

    shape1 = arr1.shape
    shape2 = arr2.shape

    for i in range(mdim):
        axis1_i = shape1[i]
        axis2_i = shape2[i]
        if axis1_i == 1 and axis2_i > 1:
            arr1 = repeat(arr1, i, axis2_i)
        elif axis1_i > 1 and axis2_i == 1:
            arr2 = repeat(arr2, i, axis1_i)
        elif axis1_i != axis2_i:
            raise IncompatibleDimensions
        # print(arr1)

        # print('\n\n\n')
    return arr1, arr2


# t0 = expand_dims(myslice, arr)
shape = [1, 2, 3, 4]
size = reduce(lambda x, y: x*y, shape)
xx = arange(size).reshape(shape)
y = arange(2).reshape([2, 1, 1, 1]) + 2


raxes = [0, 1, 2]
repts = [2, 3, 1]


def super_repeat(arr, raxes, repts):
    def recurse(ix):
        if ix == 0:
            # print(raxes[0], repts[0])
            repeat_nocopy(arr, raxes[0], 6)
        else:
            for i in range(repts[ix]):
                print(ix)
                recurse(ix - 1)
    recurse(len(raxes) - 1)


# super_repeat(xx, raxes, repts)
print(y)
# repeat_nocopy(xx, [0, 2], [2, 4])
repeat_nocopy(y, [1, 2, 3], [2, 3, 4])


# v = repeat(y, 3, 5)
# print(xx)
# print(v.shape)

# print(xx)
# print('\n')
# print(y)
# print('\n')

xx, y = broadcast_copy(xx, y)
print(y)
# v = xx + y
# print(v)

# print(v.shape)

# print(xx + y)

# repeat_nocopy(xx, 0, 4)

# xx, y = broadcast_copy(xx, y)
