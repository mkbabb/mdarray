def get_strides(shape):
    mdim = len(shape)
    init = 1
    strides = [0]*mdim
    strides[0] = init

    for i in range(mdim - 1):
        init *= shape[i]
        strides[i+1] = init

    return strides


def iter_array(shape, strides):
    mdim = len(shape)
    raxes = [0, 1, 2]
    repts = [0, 0, 2]

    ndim = len(raxes)
    counters = [0]*mdim
    rcounters = [0]*ndim

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

                add += counters[n]*strides[n]
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
#                     ix3 = pair_wise_accumulate(axis_counter, strides)

#                     try:
#                         a_val = data[ix3]
#                     except:
#                         a_val = nan

#                     arr_out.data[j] = a_val
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
