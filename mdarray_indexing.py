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
		self.shape = [1]
		if isinstance(slice_array, gslice):
			self.slice_array = slice_array.slice_array
		else:
			slc_flat, slc_mdim, slc_shape = flatten(slice_array)
			M = len(slc_shape)

			if slc_mdim == 1:
				if M <= a_inqry.mdim:
					self.slice_array = [_gslice(slice_array, a_inqry).get_slice()]
				else:
					raise TypeError
			else:
				tmp0 = []
				tmp1 = []
				new_shape = [0]*a_inqry.mdim

				for n, i in enumerate(slice_array):
					if isinstance(i, gslice):
						tmp0 += [i]
					else:
						if isinstance(i, list):
							new_shape[n] = len(i)
						elif i == nan:
							new_shape[n] = a_inqry.shape[n]
						tmp1 += [i]

				new_shape = [i for i in new_shape if i != 0]

				if len(new_shape) == a_inqry.mdim:
					self.shape = new_shape

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


def iter_axis(a, gslice_array):
	data = a.data
	a_out = [0]*1000

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
