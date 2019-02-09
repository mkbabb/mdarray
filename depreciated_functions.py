def get_strides(shape):
    mdim = len(shape)
    init = 1
    strides = [0] * mdim
    strides[0] = init

    for i in range(mdim - 1):
        init *= shape[i]
        strides[i + 1] = init

    return strides


def iter_array(shape, strides):
    mdim = len(shape)
    raxes = [0, 1, 2]
    repts = [0, 0, 2]

    ndim = len(raxes)
    counters = [0] * mdim
    rcounters = [0] * ndim

    add = 0
    ix = 0
    j = 0

    while (counters[mdim - 1] < shape[mdim - 1]):
        print(counters)
        j += 1

        counters[0] += 1

        if (counters[0] == shape[0]):
            print(counters)
            add = 0

            for n in range(mdim):
                if counters[n] == shape[n]:
                    if raxes[n] == n:
                        if rcounters[n] == repts[n]:
                            rcounters[n] = 0
                            if n != ndim - 1:
                                counters[n + 1] += 1
                            counters[n] = 0
                        else:
                            counters[n] = 0
                            rcounters[n] += 1

                add += counters[n] * strides[n]
            ix = add

        else:
            ix += strides[0]


# shape = [1, 2, 3]
# strides = get_strides(shape)

# iter_array(shape, strides)


'''
Depreciated meshgrid:
Uses expand dims and slice array, which is far more costly than simply using repeat axis.
'''

# def meshgrid(*seq):
# 	mdim = len(seq)
# 	shape = [0]*(mdim)
#
# 	counter = 0
# 	for i in range(mdim):
# 		shape[i] = len(seq[i])
# 		counter += 1
#
# 	ixs = [
# 		[i for i in range(shape[0])],
# 		]
# 	md = mdarray(shape)
# 	a_inqry = mdarray_inquery(md)
#
# 	grid = expand_slice_array(ixs, a_inqry)
# 	shape += [counter]
# 	md = tomdarray(grid).reshape(shape).T(0, counter)
# 	return md


'''
Depreciated repeat:
Uses expand dims and slice array and gslice,
which is ludicrously inefficient in contrast to repeat's current implementation.

Note: this function does NOT work properly.
'''

#
# def repeat(a, rpeat, axis):
#     a_inqry = mdarray_inquery(a)
#     mdim = a.mdim
#     slc = [0]*mdim
#     new_shape = list(a_inqry.shape)
#     new_shape[axis] *= rpeat
#
#     for i in range(mdim):
#         slc[i] = inf if i == axis else nan
#     slc = [nan, inf]
#
#     slc = gslice(slc, a_inqry)
#     md = tomdarray(iter_axis(a, slc, 12, rpeat)).reshape(new_shape)
#     return md


'''
Depreciated repeat (again):
Uses similar logic to repeat's current implementation,
but is not capable of multiple repeats along multiple axes.
'''

# def repeat(arr, raxis, rept):
#     global j
#     mdim = arr.mdim

#     new_shape = list(arr.shape)
#     new_shape[raxis] *= rept

#     arr_out = zeros(shape=new_shape)
#     axis_counter = [0]*mdim

#     raxis_s = 1 if 0 != raxis else rept

#     def recurse(ix):
#         global j

#         shape = arr.shape
#         strides = arr.strides
#         data = arr.data
#         axis = shape[ix]

#         remaining_axes = mdim - ix

#         if remaining_axes == mdim:
#             for i in range(axis):
#                 for k in range(raxis_s):
#                     axis_counter[0] = i
#                     ix_i = pair_wise_accumulate(axis_counter, strides)

#                     try:
#                         arr_val = data[ix_i]
#                     except:
#                         arr_val = nan

#                     arr_out.data[j] = arr_val
#                     j += 1
#         else:
#             for i in range(axis):
#                 axis_counter[ix] = i
#                 if ix == raxis:
#                     for k in range(rept):
#                         recurse(ix - 1)
#                 else:
#                     recurse(ix - 1)
#     j = 0
#     recurse(mdim - 1)
#     return arr_out

'''
Much more straightforward implmentation of generalised broadcasting,
but is orders of magnitude more memory-expensive.
Don't use!
'''


# def broadcast_copy(arr1, arr2):
#     mdim1 = arr1.mdim
#     mdim2 = arr2.mdim
#     shape1 = arr1.shape
#     shape2 = arr2.shape

#     mdim = mdim1

#     if mdim1 > mdim2:
#         arr2.reshape(shape2 + [1]*(mdim1 - mdim2))
#     elif mdim1 < mdim2:
#         arr1.reshape(shape1 + [1]*(mdim2 - mdim1))
#         mdim = mdim2

#     shape1 = arr1.shape
#     shape2 = arr2.shape

#     repts1 = [0]*mdim
#     repts2 = [0]*mdim

#     for i in range(mdim):
#         axis1_i = shape1[i]
#         axis2_i = shape2[i]
#         if axis1_i == 1 and axis2_i > 1:
#             arr1 = repeat(arr1, i, axis2_i)
#         elif axis1_i > 1 and axis2_i == 1:
#             arr2 = repeat(arr2, i, axis1_i)
#         elif axis1_i != axis2_i:
#             raise IncompatibleDimensions
#     return arr1, arr2

'''
Depreciated gslice implmentation.
Replaced now by ix_ and slice_array.
'''


# class _gslice(object):
#     def __init__(self, slice_array, a_inqry):
#         self.slice_array = slice_array

#         self.mdim = a_inqry.mdim
#         self.strides = a_inqry.strides
#         self.shape = a_inqry.shape

#         self.arg_axis = 0

#         if len(self.slice_array) == 0:
#             self.slice_array = [0]

#         mdim = len(self.slice_array)

#         if mdim < self.mdim:
#             self.slice_array += [nan] * (self.mdim - mdim)
#         elif mdim > self.mdim:
#             self.slice_array = self.slice_array[:self.mdim]

#     def get_slice(self):
#         new_shape = []
#         new_strides = []
#         new_size = 1

#         for n, i in enumerate(self.slice_array):
#             if i != nan:
#                 tmp = i * self.strides[n]
#                 self.arg_axis += tmp
#                 self.slice_array[n] = tmp
#                 self.mdim -= 1
#             else:
#                 tmp = self.shape[n]
#                 new_size *= tmp
#                 new_shape += [tmp]
#                 new_strides += [self.strides[n]]

#         if len(new_shape) == 0:
#             self.mdim = 1
#             self.shape = [1]
#             self.strides = [1]
#             self.size = 1
#         else:
#             self.shape = new_shape
#             self.strides = new_strides
#             self.size = new_size

#         return self

#     def __repr__(self):
#         return str(self.slice_array)


# class gslice(object):
#     def __init__(self, slice_array, arr):
#         tmp0 = []
#         tmp1 = []
#         new_shape = [0] * arr.mdim

#         for n, i in enumerate(slice_array):
#             if i == inf:
#                 slice_array[n] = [j for j in range(arr.shape[n])]

#         for n, i in enumerate(slice_array):
#             if isinstance(i, gslice):
#                 tmp0 += [i]
#             else:
#                 tmp1 += [i]
#                 if len(slice_array) == arr.mdim:
#                     pass

#         new_shape = [i for i in new_shape if i != 0]

#         self.shape = new_shape if len(new_shape) <= a_inqry.mdim else [1]

#         tmp1 = remove_extraneous_dims(tmp1)
#         self.slice_array = make_iter_list(tmp1)

#         for n, i in enumerate(self.slice_array):
#             self.slice_array[n] = _gslice(i, a_inqry).get_slice()

#         for i in tmp0:
#             self.slice_array += [i.slice_array[0]]

#     def __repr__(self):
#         if len(self.slice_array) == 1:
#             return str(self.slice_array[0])
#         return str(self.slice_array)


# def expand_slice_array(slice_array, mdim, strides=None):
#     ndim = len(slice_array)

#     if strides:
#         pad_length = strides[ndim - 1] if mdim != 0 else 1
#     else:
#         pad_length = 1

#     broadcast_length = len(slice_array[0])
#     out_array = [[0] * mdim] * (broadcast_length * pad_length)

#     print(broadcast_length, pad_length)

#     tmp0 = [0] * mdim
#     l = 0
#     for i in range(broadcast_length):
#         for j in range(ndim):
#             tmp0[j] = slice_array[j][i]

#         if pad_length == 1:
#             out_array[i] = list(tmp0)
#         else:
#             for k in range(pad_length):
#                 tmp0[mdim - 1] = k
#                 out_array[l] = list(tmp0)
#                 l += 1

#     return out_array


# def expand_dims(slice_array, arr):
#     ndim = len(slice_array)
#     mdim = arr.mdim

#     broadcast_length = 0
#     lens = 0
#     for i in range(ndim):
#         arr_i = slice_array[i]
#         try:
#             ndim_i = len(arr_i)
#         except TypeError:
#             if arr_i == inf or arr_i == Ellipsis:
#                 slice_array[i] = [j for j in range(arr.shape[i])]
#                 ndim_i = arr.shape[i]
#             else:
#                 slice_array[i] = [arr_i]
#                 ndim_i = 1

#         lens += ndim_i
#         broadcast_length = ndim_i if ndim_i > broadcast_length else broadcast_length

#     broadcast_length = mdim if broadcast_length > mdim else broadcast_length

#     if lens // ndim == ndim:
#         return slice_array

#     tmp0 = [0] * broadcast_length
#     for i in range(ndim):
#         ndim_i = len(slice_array[i])

#         ndim_i = ndim_i if ndim_i < mdim else mdim
#         pad_length = broadcast_length - ndim_i

#         j = 0
#         while j < ndim_i:
#             tmp0[j] = slice_array[i][j]
#             j += 1

#         if pad_length > 0:
#             while j < ndim_i + pad_length:
#                 tmp0[j] = slice_array[i][ndim_i - 1]
#                 j += 1

#         slice_array[i] = list(tmp0)

#     return slice_array


'''
Iteration over an array by a given gslice. Used for any type of indexing into an m-d array.
'''


# def iter_gslice(arr, gslice_array, size):
#     data = arr.data
#     a_out = [0] * size

#     def recurse(g, axis_counter, ix, j):
#         axis = g.shape[ix]
#         remaining_axes = g.mdim - ix

#         if remaining_axes == 1:
#             for i in range(axis):
#                 axis_counter[g.mdim - 1] = i
#                 ix_i = pair_wise_accumulate(
#                     axis_counter, g.strides) + g.arg_axis

#                 arr_val = data[ix_i]
#                 a_out[j] = arr_val
#                 j += 1

#         else:
#             for i in range(axis):
#                 axis_counter[ix] = i
#                 j = recurse(g, axis_counter, ix + 1, j)
#         return j

#     j = 0
#     for i in gslice_array.slice_array:
#         axis_counter = [0] * i.mdim
#         ix = 0

#         j = recurse(i, axis_counter, ix, j)

#     return a_out

# def make_iter_list(slice_array):
#     if len(slice_array) == 0:
#         return slice_array

#     array_out = []

#     def recurse(slice_array, array_out):
#         flag = False
#         recursive_flag = False
#         tmp0 = []

#         for n, i in enumerate(slice_array):
#             if isinstance(i, list) and not recursive_flag:
#                 for j in slice_array[n]:
#                     tmp1 = [k for m, k in enumerate(slice_array) if m != n]
#                     tmp1.insert(n, j)

#                     array_out = recurse(tmp1, array_out)
#                     recursive_flag = True
#                 flag = True

#             else:
#                 tmp0 += [i]

#         if not flag:
#             array_out += [tmp0]

#         return array_out

#     return recurse(slice_array, array_out)
'''
Depreciated padding routine:
'''
# def pad_array(arr, pad_width, pad_func=None):
#     if pad_func == None:
#         pad_func = lambda shape, arr_i, i, j: full(shape=shape, fill_value=0)

#     ndim = len(pad_width)
#     pdim = len(pad_width[0])

#     shape = arr.shape
#     new_shape = list(shape)

#     for i in range(ndim):
#         v = reduce(lambda x, y: x + y, pad_width[i])
#         new_shape[i] += v

#     arrs = [0] * (pdim + 1)
#     middle = len(arrs) // 2
#     shape_i = list(shape)
#     arr_i = arr

#     for i in range(ndim):
#         pad_i = pad_width[i]

#         for j in range(pdim):
#             shape_ij = list(shape_i)
#             shape_ij[i] = pad_i[j]

#             arr_ij = pad_func(shape=shape_ij, arr_i=arr_i, i=i, j=j)
#             arrs[j] = arr_ij

#         arrs[pdim] = arr_i
#         swap_item(arrs, middle, pdim)

#         arr_i = concatenate(*arrs, caxis=i)
#         shape_i[i] = new_shape[i]

#     return arr_i
