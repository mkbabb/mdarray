from functools import reduce
from mdarray_helper import pair_wise_accumulate, get_strides
from mdarray_types import inf, nan, mdarray_inquery


class _gslice(object):
	def __init__(self, slice_array, a_inqry):
		self.slice_array = slice_array

		self.mdim = a_inqry.mdim
		self.strides = a_inqry.strides
		self.shape = a_inqry.shape

		self.arg_axis = 0

		if len(self.slice_array) == 0:
			self.slice_array = [0]

		mdim = len(self.slice_array)

		if mdim < self.mdim:
			self.slice_array += [nan]*(self.mdim - mdim)
		elif mdim > self.mdim:
			self.slice_array = self.slice_array[:self.mdim]

	def get_slice(self):
		new_shape = []
		new_strides = []
		new_size = 1

		for n, i in enumerate(self.slice_array):
			if i != nan:
				tmp = i*self.strides[n]
				self.arg_axis += tmp
				self.slice_array[n] = tmp
				self.mdim -= 1
			else:
				tmp = self.shape[n]
				new_size *= tmp
				new_shape += [tmp]
				new_strides += [self.strides[n]]

		if len(new_shape) == 0:
			self.mdim = 1
			self.shape = [1]
			self.strides = [1]
			self.size = 1
		else:
			self.shape = new_shape
			self.strides = new_strides
			self.size = new_size

		return self

	def __repr__(self):
		return str(self.slice_array)


class gslice(object):
	def __init__(self, slice_array, a_inqry):
		if isinstance(slice_array, gslice):
			self.slice_array = slice_array.slice_array
			self.shape = slice_array.shape
		else:
			tmp0 = []
			tmp1 = []
			new_shape = [0]*a_inqry.mdim

			for n, i in enumerate(slice_array):
				if i == inf:
					slice_array[n] = [j for j in range(a_inqry.shape[n])]

			for n, i in enumerate(slice_array):
				if isinstance(i, gslice):
					tmp0 += [i]
				else:
					tmp1 += [i]
					if len(slice_array) == a_inqry.mdim:
						pass

			new_shape = [i for i in new_shape if i != 0]

			self.shape = new_shape if len(new_shape) <= a_inqry.mdim else [1]

			tmp1 = remove_extraneous_dims(tmp1)
			self.slice_array = make_iter_list(tmp1)

			for n, i in enumerate(self.slice_array):
				self.slice_array[n] = _gslice(i, a_inqry).get_slice()

			for i in tmp0:
				self.slice_array += [i.slice_array[0]]

	def __repr__(self):
		if len(self.slice_array) == 1:
			return str(self.slice_array[0])
		return str(self.slice_array)


def iter_axis(a, gslice_array, size, repeat=0, raxis=0):
	data = a.data
	a_out = [0]*size

	def recurse(g, ix1, ix2, j):
		axis = g.shape[ix2]
		remaining_axes = g.mdim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				ix1[g.mdim - 1] = i
				ix3 = pair_wise_accumulate(ix1, g.strides) + g.arg_axis

				a_val = data[ix3]
				a_out[j] = a_val
				j += 1

		else:
			for i in range(axis):

				ix1[ix2] = i
				j = recurse(g, ix1, ix2 + 1, j)
				print(a_out)
		return j

	j = 0
	for i in gslice_array.slice_array:
		ix1 = [0]*i.mdim
		ix2 = 0

		j = recurse(i, ix1, ix2, j)

	return a_out


def remove_extraneous_dims(a):
	def recurse(a):
		if len(a) == 1:
			try:
				a = recurse(a[0])
				return a
			except IndexError:
				return a
		else:
			return a

	return recurse(a)


def make_iter_list(slice_array):
	if len(slice_array) == 0:
		return slice_array

	array_out = []

	def recurse(slice_array, array_out):
		flag = False
		recursive_flag = False
		tmp0 = []

		for n, i in enumerate(slice_array):
			if isinstance(i, list) and not recursive_flag:
				for j in slice_array[n]:
					tmp1 = [k for m, k in enumerate(slice_array) if m != n]
					tmp1.insert(n, j)

					array_out = recurse(tmp1, array_out)
					recursive_flag = True
				flag = True

			else:
				tmp0 += [i]

		if not flag:
			array_out += [tmp0]

		return array_out

	return recurse(slice_array, array_out)


def flatten(a, order=1):
	global shape, dim_counter
	shape = [len(a)]
	dim_counter = 0

	def recurse(a):
		global shape, dim_counter
		ndim = len(a)

		tmp = []
		dim_counter = 0

		for i in range(ndim):
			a_i = a[i]

			if isinstance(a_i, list):
				tmp0 = recurse(a_i)
				M = len(a_i)

				if len(shape) <= dim_counter + 1:
					shape.insert(1, M)

				dim_counter += 1
				tmp += [tmp0] if dim_counter <= order else tmp0
			else:
				tmp += [a_i]

		return tmp

	flt = recurse(a)
	return flt, dim_counter, shape


def make_nested(a):
	data = a.data
	mdim = a.mdim
	shape = a.shape
	strides = a.strides

	ix1 = [0]*mdim
	ix2 = 0

	def recurse(ix1, ix2):
		global j
		tmp = []

		axis = shape[ix2]
		remaining_axes = mdim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				ix1[mdim - 1] = i
				ix3 = pair_wise_accumulate(ix1, strides)

				a_val = data[ix3]
				tmp += [a_val]
		else:
			for i in range(axis):
				ix1[ix2] = i
				tmp += [recurse(ix1, ix2 + 1)]

		return tmp

	return recurse(ix1, ix2)
