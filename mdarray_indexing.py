from functools import reduce
from mdarray_helper import pair_wise_accumulate


class md_nan(object):
	def __init__(self):
		self.value = 'not a number!'

	def __repr__(self):
		return 'nan'


nan = md_nan()


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
	def __init__(self, a, slice_array):
		super(gslice, self).__init__(a)

		mdim = len(slice_array)

		if mdim != len(self.strides):
			slice_array += [nan]*(mdim - len(self.strides) - 1)
		#
		# shape = a_inqry.shape
		# strides = a_inqry.strides
		#
		# self.slice_array = []
		# self.shape = []
		# self.strides = []
		# self.mdim = 0
		#
		# for n, i in enumerate(slice_array):
		# 	if i != md_nan:
		# 		self.slice_array += [i*strides[n]]
		# 	else:
		# 		self.mdim += 1
		# 		self.shape += [shape[n]]
		# 		self.strides += [strides[n]]
		#
		# if len(self.shape) == 0:
		# 	self.mdim = 1
		# 	self.shape = [1]
		# 	self.strides = [1]
		#
		# if len(self.slice_array) == 0:
		# 	self.slice_array = [0]
		# 	self.shape = shape
		# 	self.strides = strides
		# 	self.mdim = a_inqry.mdim
		# 	self.size = a_inqry.size
		# 	self.arg_axis = 0
		# else:
		# 	self.size = reduce(lambda x, y: x*y, self.shape)
		# 	self.arg_axis = reduce(lambda x, y: x + y, self.slice_array)




j = 0


def _iter_axis(a, _gslice):
	global j

	data = a.data
	mdim = _gslice.mdim
	shape = _gslice.shape
	strides = _gslice.strides
	size = _gslice.size
	arg_axis = _gslice.arg_axis

	print(size, shape)

	array_out = [0]*size

	ix1 = [0]*mdim
	ix2 = 0
	j = 0

	def recurse(ix1, ix2):
		global j
		axis = shape[ix2]
		remaining_axes = mdim - ix2

		if remaining_axes == 1:
			for i in range(axis):
				ix1[mdim - 1] = i
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
	shape = a_inqry.shape
	new_shape = [0]*len(_pgslice)

	for n, i in enumerate(_pgslice):
		if isinstance(i, list):
			new_shape[n] = len(i)
		elif i == md_nan:
			new_shape[n] = shape[n]

	new_shape = [i for i in new_shape if i != 0]

	if isinstance(_pgslice, list):
		_gslice_list = make_iter_list(_pgslice, [])

		for n, i in enumerate(_gslice_list):
			_gslice_list[n] = gslice(i, a_inqry)

	elif isinstance(_pgslice, gslice):
		_gslice_list = [_pgslice]

	else:
		_gslice_list = [gslice(_pgslice, a_inqry)]

	return new_shape, _gslice_list
