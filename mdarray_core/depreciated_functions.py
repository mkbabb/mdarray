def get_strides(shape):
    strides = []
    tmp = 1
    for i in shape:
        strides += [tmp]
        tmp *= i

    return strides


def iter_array(shape, strides):
    N = len(shape)
    counters = [0]*N

    add = 0
    ix = 0

    while (counters[0] < shape[0]):
        counters[N - 1] += 1

        if (counters[N - 1] == shape[N - 1]):
            add = 0

            for n in range(N - 1, -1, -1):
                if counters[n] == shape[n] and n != 0:
                    counters[n - 1] += 1
                    counters[n] = 0

                add += counters[n]*strides[n]
            ix = add

        else:
            ix += strides[N - 1]


'''
Depreciated meshgrid:
Uses expand dims and slice array, which is far more costly than simply using repeat axis.
'''

# def meshgrid(*seq):
# 	N = len(seq)
# 	shape = [0]*(N)
#
# 	counter = 0
# 	for i in range(N):
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
