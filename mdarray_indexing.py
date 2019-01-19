from functools import reduce
from mdarray_helper import pair_wise_accumulate, get_strides
from mdarray_types import inf, nan


class mdarray_inquery(object):
	def __init__(self, a):
		self.mdim = a.mdim
		self.shape = a.shape
		self.size = a.size
		self.strides = a.strides
		self.dtype = a.dtype

	def __str__(self):
		s = ''

		max_len = 0
		for i in self.__dict__.keys():
			current_len = len(str(i)) + 1

			if current_len > max_len:
				max_len = current_len

		print(max_len)
		for i, j in self.__dict__.items():
			current_len = len(str(i))
			space = ' '*(max_len - current_len)
			s += '{0}:{1}{2}\n'.format(i, space, j)
		return s


class gslice(mdarray_inquery):
	def __init__(self, slice_array, a=None):
		self.slice_array = slice_array

		if len(self.slice_array) == 0:
			self.slice_array = [0]

		self.arg_axis = 0
		mdim = len(slice_array)

		if not a:
			self.mdim = mdim
			self.shape = [1]
			self.strides = [1]
			self.size = 1
		else:
			self.a = a
			super(gslice, self).__init__(a)

		if mdim < self.mdim:
			self.slice_array += [nan]*(self.mdim - mdim)
		elif mdim > self.mdim:
			self.slice_array = self.slice_array[:self.mdim]

	def get_slice(self):
		new_shape = []
		new_strides = []
		new_size = 1
		arg_axis = 0

		for n, i in enumerate(self.slice_array):
			if i != nan:
				tmp = i*self.strides[n]
				arg_axis += tmp
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

		self.arg_axis = arg_axis

		return self

	def __repr__(self):
		return str(self.strides)


def generate_gslice_list(a, slice_arrays):
	gslice_list = []
	for i in slice_arrays:
		gslice_list += [gslice(i, a).get_slice()]
	return gslice_list


def iter_axis(a, _gslice):
	data = a.data
	mdim = _gslice.mdim
	shape = _gslice.shape
	strides = _gslice.strides
	size = _gslice.size
	arg_axis = _gslice.arg_axis

	a_out = [0]*size

	ix1 = [0]*mdim
	ix2 = 0

	def recurse(ix1, ix2, j):
		axis = shape[ix2]
		remaining_axes = mdim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				ix1[mdim - 1] = i
				ix3 = pair_wise_accumulate(ix1, strides) + arg_axis

				a_val = data[ix3]
				a_out[j] = a_val
				j += 1

		else:
			for i in range(axis):
				ix1[ix2] = i
				j = recurse(ix1, ix2 + 1, j)
		return j

	recurse(ix1, ix2, 0)
	return a_out


def _slice_array(a, _gslice):
	out_array = []

	def recurse(_gslice, out_array):
		if isinstance(_gslice, list):
			for i in _gslice:
				out_array = recurse(i, out_array)

		elif isinstance(_gslice, gslice):
			tmp = iter_axis(a, _gslice)
			out_array += tmp

		return out_array

	return recurse(_gslice, out_array)


def make_iter_list(slice_array):
	if len(slice_array) == 0:
		return slice_array
	array_out = []

	def recursive(slice_array, array_out):
		flag = False
		recursive_flag = False
		tmp0 = []

		for n, i in enumerate(slice_array):
			if isinstance(i, list) and not recursive_flag:
				for j in slice_array[n]:
					tmp1 = [k for m, k in enumerate(slice_array) if m != n]
					tmp1.insert(n, j)

					array_out = recursive(tmp1, array_out)
					recursive_flag = True
				flag = True
			else:
				tmp0 += [i]

		if not flag:
			array_out += [tmp0]

		return array_out

	return recursive(slice_array, array_out)


def make_array_indicies(a, pgslice):
	if isinstance(pgslice, gslice):
		return pgslice
	else:
		gslice_list = []
		slice_arrays = []

		for n, i in enumerate(pgslice):
			if isinstance(i, gslice):
				gslice_list += [i]
			else:
				print(i)
				slice_arrays += [i]

		if len(slice_arrays) > 0:
			slice_arrays = make_iter_list(slice_arrays)

		elif len(slice_arrays) == 0 and len(gslice_list) == 0:
			slice_arrays = [pgslice]

		for i in slice_arrays:
			gslice_list += [gslice(i, a).get_slice()]

	return gslice_list


def flatten(a):
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

				tmp += tmp0
			else:
				tmp += [a_i]
		return tmp, dim_counter, shape

	return recurse(a, 1, shape)


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
