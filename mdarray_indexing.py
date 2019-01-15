from functools import reduce
from mdarray_helper import pair_wise_accumulate


class nan:
	value = 'not a number!'


class gslice(object):
	def __init__(self, slice_array, a_inqry):

		M = len(slice_array)

		if M != len(a_inqry.strides):
			slice_array += [nan]*(M - len(a_inqry.strides) - 1)

		shape = a_inqry.shape
		strides = a_inqry.strides

		self.slice_array = []
		self.shape = []
		self.strides = []
		self.dim = 0

		for n, i in enumerate(slice_array):
			if i != nan:
				self.slice_array += [i*strides[n]]
			else:
				self.dim += 1
				self.shape += [shape[n]]
				self.strides += [strides[n]]

		if len(self.shape) == 0:
			self.dim = 1
			self.shape = [1]
			self.strides = [1]

		if len(self.slice_array) == 0:
			self.slice_array = [0]
			self.shape = shape
			self.strides = strides
			self.dim = a_inqry.dim
			self.size = a_inqry.size
			self.arg_axis = 0
		else:
			self.size = reduce(lambda x, y: x*y, self.shape)
			self.arg_axis = reduce(lambda x, y: x + y, self.slice_array)


j = 0


def _iter_axis(a, _gslice):
	global j

	data = a.data
	dim = _gslice.dim
	shape = _gslice.shape
	strides = _gslice.strides
	size = _gslice.size
	arg_axis = _gslice.arg_axis

	print(size, shape, strides, a.strides)

	array_out = [0]*size

	ix1 = [0]*dim
	ix2 = 0
	j = 0

	def recurse(ix1, ix2):
		global j
		axis = shape[ix2]
		remaining_axes = dim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				ix1[dim - 1] = i
				ix3 = pair_wise_accumulate(ix1, strides) + arg_axis

				a_val = data[ix3]
				array_out[j] = a_val
				j += 1
		else:
			for i in range(axis):
				ix1[ix2] = i
				recurse(ix1, ix2 + 1)
		return array_out

	return recurse(ix1, ix2)


def _slice_array(a, _gslice):
	out_array = []

	def recurse(_gslice, out_array):
		if isinstance(_gslice, list):
			for i in _gslice:
				out_array = recurse(i, out_array)
		elif isinstance(_gslice, gslice):
			tmp = _iter_axis(a, _gslice)
			out_array += tmp

		return out_array

	return recurse(_gslice, out_array)


def make_iter_list(slice_array, out_indicies):
	tmp0 = []

	flag = False
	recur_flag = False

	for n, i in enumerate(slice_array):
		if isinstance(i, list) and not recur_flag:

			for j in slice_array[n]:
				tmp1 = [k for m, k in enumerate(slice_array) if m != n]
				tmp1.insert(n, j)

				out_indicies = make_iter_list(tmp1, out_indicies)

				recur_flag = True
			flag = True

		else:
			tmp0 += [i]

	if not flag:
		out_indicies += [tmp0]

	return out_indicies


def make_gslice_list(_pgslice, a_inqry, ):
	N = len(_pgslice)
	new_shape = [3, 2, 1]
	# new_shape = [0]*N
	#
	# shape = a_inqry.shape
	#
	# for n, i in enumerate(_pgslice):
	# 	if isinstance(i, list):
	# 		new_shape[n] = len(i)
	# 	elif i != nan and i > 0:
	# 		new_shape[n] = shape[n]

	if isinstance(_pgslice, list):
		_gslice_list = make_iter_list(_pgslice, [])

		for n, i in enumerate(_gslice_list):
			_gslice_list[n] = gslice(i, a_inqry)

	elif isinstance(_pgslice, gslice):
		_gslice_list = [_pgslice]

	else:
		_gslice_list = [gslice(_pgslice, a_inqry)]

	return new_shape, _gslice_list
