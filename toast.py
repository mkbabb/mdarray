from mdarray_helper import get_strides, update_dict
from mdarray_indexing import gslice, iter_axis
from mdarray import mdarray, arange, tomdarray
from mdarray_types import nan, inf, mdarray_inquery
from functools import reduce
import numpy as np


def flatten(a, order=1):
	shape = [len(a)]

	def recurse(a, dim_counter, shape):
		N = len(a)

		tmp = []
		dim_counter = 1

		for i in range(N):
			a_i = a[i]

			if isinstance(a_i, list):
				tmp0, dim_counter, shape = recurse(a_i, dim_counter, shape)
				M = len(a_i)

				if len(shape) == dim_counter:
					shape += [M]
				else:
					shape[dim_counter] = M

				dim_counter += 1

				tmp += [tmp0] if dim_counter <= order else tmp0

			else:
				tmp += [a_i]
		return tmp, dim_counter, shape

	return recurse(a, 1, shape)


def expand_slice_array(slice_array, a_inqry):
	mdim = a_inqry.mdim
	ndim = len(slice_array)

	pad_length = a_inqry.strides[ndim - 1] if mdim != ndim else 1
	broadcast_length = len(slice_array[0])

	out_array = [[0]*mdim]*(broadcast_length*pad_length)

	tmp0 = [0]*mdim
	l = 0
	for i in range(broadcast_length):
		for j in range(ndim):
			tmp0[j] = slice_array[j][i]

		if pad_length == 1:
			out_array[i] = list(tmp0)
		else:
			for k in range(pad_length):
				tmp0[mdim - 1] = k
				out_array[l] = list(tmp0)
				l += 1

	return out_array


def expand_dims(slice_array, a_inqry):
	ndim = len(slice_array)
	mdim = a_inqry.mdim

	broadcast_length = 0
	lens = 0
	for i in range(ndim):
		a_i = slice_array[i]
		try:
			ndim_i = len(a_i)
		except TypeError:
			if a_i == inf or a_i == Ellipsis:
				slice_array[i] = [j for j in range(a_inqry.shape[i])]
				ndim_i = a_inqry.shape[i]
			else:
				slice_array[i] = [a_i]
				ndim_i = 1

		lens += ndim_i
		broadcast_length = ndim_i if ndim_i > broadcast_length else broadcast_length

	broadcast_length = mdim if broadcast_length > mdim else broadcast_length

	if lens//ndim == ndim:
		return slice_array

	tmp0 = [0]*broadcast_length
	for i in range(ndim):
		ndim_i = len(slice_array[i])

		ndim_i = ndim_i if ndim_i < mdim else mdim
		pad_length = broadcast_length - ndim_i

		j = 0
		while j < ndim_i:
			tmp0[j] = slice_array[i][j]
			j += 1

		if pad_length > 0:
			while j < ndim_i + pad_length:
				tmp0[j] = slice_array[i][ndim_i - 1]
				j += 1

		slice_array[i] = list(tmp0)

	return slice_array


def meshgrid(*seq):
	N = len(seq)
	shape = [0]*(N)

	counter = 0
	for i in range(N):
		shape[i] = len(seq[i])
		counter += 1

	ixs = [
		[i for i in range(shape[0])],
		]
	md = mdarray(shape)
	a_inqry = mdarray_inquery(md)

	grid = expand_slice_array(ixs, a_inqry)
	shape += [counter]
	md = tomdarray(grid).reshape(shape).T(0, counter)
	return md


def repeat(a, rpeat, axis):
	a_inqry = mdarray_inquery(a)
	mdim = a.mdim
	slc = [0]*mdim
	new_shape = list(a_inqry.shape)
	new_shape[axis] *= rpeat

	for i in range(mdim):
		slc[i] = inf if i == axis else nan
	slc = [nan, inf]

	slc = gslice(slc, a_inqry)
	md = tomdarray(iter_axis(a, slc, 12, rpeat)).reshape(new_shape)
	return md


# grid = np.meshgrid(range(0, 5), range(0, 5), range(0, 2))
# print(grid)
#
# grid2 = meshgrid(range(0, 5), range(0, 5), range(0, 2))
# grid2.set_print_formatter(lambda x: "{0}".format(x))
# print(grid2.shape)


a = [[[1, 2, 3],
	  [4, 5, 6]],

     [[1, 2, 3],
	  [4, 5, 6]]]




md = tomdarray(a)
print(md)
b = np.asarray(a)
print(b.shape)
b = b.repeat(2, axis=2)
print(b)
print(b.shape)
print(b.flatten())



a_inqry = mdarray_inquery(md)
print(a_inqry)

gslc = gslice([nan, nan, inf], a_inqry)


v = iter_axis(md, gslc, 24, 2, 3)

md = tomdarray(v).reshape([2, 2, 6])
print(md)





