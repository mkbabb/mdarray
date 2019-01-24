from functools import reduce

import numpy as np

from mdarray import arange, mdarray, tomdarray
from mdarray_formatting import pad_array_fmt
from mdarray_helper import get_strides, pair_wise_accumulate, update_dict, swap_item
from mdarray_indexing import gslice, iter_axis, make_nested
from mdarray_types import inf, mdarray_inquery, nan


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


def expand_slice_array(slice_array, arr_inqry):
	mdim = arr_inqry.mdim
	ndim = len(slice_array)

	pad_length = arr_inqry.strides[ndim - 1] if mdim != ndim else 1
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


def expand_dims(slice_array, arr_inqry):
	ndim = len(slice_array)
	mdim = arr_inqry.mdim

	broadcast_length = 0
	lens = 0
	for i in range(ndim):
		arr_i = slice_array[i]
		try:
			ndim_i = len(arr_i)
		except TypeError:
			if arr_i == inf or arr_i == Ellipsis:
				slice_array[i] = [j for j in range(arr_inqry.shape[i])]
				ndim_i = arr_inqry.shape[i]
			else:
				slice_array[i] = [arr_i]
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


def repeat(arr, rept, raxis):
	data = arr.data
	mdim = arr.mdim
	shape = arr.shape
	strides = arr.strides

	new_shape = list(shape)
	new_shape[raxis] *= rept

	ix1 = [0]*mdim
	ix2 = 0
	arr_out = [0]*arr.size*rept

	raxis_s = 1 if mdim - 1 != raxis else rept

	def recurse(ix1, ix2, j):
		axis = shape[ix2]
		remaining_axes = mdim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				for k in range(raxis_s):
					ix1[mdim - 1] = i
					ix3 = pair_wise_accumulate(ix1, strides)

					try:
						a_val = data[ix3]
					except:
						a_val = nan

					arr_out[j] = a_val
					j += 1
		else:
			for i in range(axis):
				ix1[ix2] = i
				if ix2 == raxis:
					for k in range(rept):
						j = recurse(ix1, ix2 + 1, j)
				else:
					j = recurse(ix1, ix2 + 1, j)

		return j

	recurse(ix1, ix2, 0)
	return tomdarray(arr_out).reshape(new_shape)


def meshgrid_md(*seq):
	seq = tuple(seq)
	lens = list(map(len, seq))
	mdim = len(seq)

	size = 1
	for i in range(mdim):
		size *= lens[i]

	arr_out = []
	for n, i in enumerate(seq):
		slc = [1]*mdim
		slc[n] = lens[n]
		arr_i = tomdarray(i).reshape(slc)
		for m, j in enumerate(lens):
			if m != n:
				arr_i = repeat(arr_i, j, raxis=m)
		arr_out.append(arr_i)

	return tuple(arr_out)


def concatenate(*seq, caxis):
	arrs = list(seq)
	arr1 = arrs[0]

	ndim = len(arrs)
	mdim = arr1.mdim

	new_shape = list(arr1.shape)
	print(new_shape)
	new_size = 0

	ixs = [[0]*mdim]*ndim
	for i in range(ndim):
		arr_i = arrs[i]

		new_size += arr_i.size
		if i > 0:
			new_shape[caxis] += arr_i.shape[caxis]

	print(new_size)

	arr_out = [0]*(new_size)

	def recurse(warr, ix2, j):
		ix1 = ixs[warr]
		arr_i = arrs[warr]

		shape = arr_i.shape
		strides = arr_i.strides
		data = arr_i.data
		axis = shape[ix2]

		remaining_axes = mdim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				ix1[mdim - 1] = i

				ix3 = pair_wise_accumulate(ix1, strides)

				try:
					a_val = data[ix3]
				except:
					a_val = nan

				arr_out[j] = a_val
				j += 1

		else:
			for i in range(axis):
				ix1[ix2] = i
				j = recurse(warr, ix2 + 1, j)

				if ix2 == caxis - 1:
					for k in range(ndim-1):
						j = recurse(k+1, ix2 + 1, j)

		return j


	j = recurse(0, 0, 0)

	return tomdarray(arr_out).reshape(new_shape)


# grid = np.meshgrid(range(0, 5), range(0, 5), range(0, 2))
# print(grid)
#
# grid2 = meshgrid(range(0, 5), range(0, 5), range(0, 2))
# grid2.set_print_formatter(lambda x: "{0}".format(x))
# print(grid2.shape)


shape = [10, 10, 2, 2]
size = reduce(lambda x, y: x*y, shape)

md = arange(size).reshape(shape)
a_inqry = mdarray_inquery(md)

# b = np.asarray(md.to_list())
# print(b.shape)
# b = b.repeat(1, axis=3)
# print(b.strides)
# print(md.strides)
#
# print(b.shape)


# print(b)


# v = meshgrid_md(range(0, 9), range(0, 9), range(0, 2))
# v = itertest(md, 1, 0)


# v.formatter = pad_array_fmt(v)
# v.T(1, 2)

# print(v[0])

# print(v[0])
# print(v[1])
# print(v[2])
#
# v2 = np.meshgrid(range(0, 9), range(0, 9), range(0, 2))
# print(v2[0].shape)

shape1 = [5, 5, 2]
size1 = reduce(lambda x, y: x*y, shape1)

arr1 = arange(size1).reshape(shape1)

print(arr1)

shape2 = [5, 5, 2]
size2 = reduce(lambda x, y: x*y, shape2)

arr2 = arange(size2).reshape(shape2)*999
print(arr2)

arr_out = concatenate(arr1, arr2, arr2,arr2, arr1, caxis=2)
print(arr_out)
