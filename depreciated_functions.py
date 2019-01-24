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
