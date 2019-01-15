import numpy as np
from functools import reduce

from mdarray_helper import pair_wise_accumulate, pair_wise, swap_item
from mdarray_indexing import _iter_axis, make_iter_list, gslice, _slice_array, nan, make_gslice_list
from mdarray_formatting import array_print


def get_strides(shape):
	N = len(shape)
	init = 1
	strides = [0]*N
	strides[N - 1] = init

	for i in range(N - 1):
		init *= shape[N - (i + 1)]
		strides[N - (i + 2)] = init

	return strides


class mdarray(object):
	def __init__(self, shape=None, size=None, data=None):
		if not shape:
			if not size:
				raise ZeroDivisionError
			self.shape = [size]
			self.size = size
		else:
			self.shape = shape
			self.size = reduce(lambda x, y: x*y, shape)

		self.mdim = len(self.shape)
		self.strides = get_strides(self.shape)

		self.data = data if data else [0]*self.size

		self.dtype = type(self.data[0])

	def reshape(self, new_shape):
		new_size = reduce(lambda x, y: x*y, new_shape)

		if new_size != self.size:
			raise ZeroDivisionError

		self.shape = new_shape

		self.get_mdim()
		self.get_strides()
		self.size = new_size

	def get_mdim(self):
		self.mdim = len(self.shape)

	def get_size(self):
		self.size = reduce(lambda x, y: x*y, self.shape)

	def get_strides(self):
		self.strides = get_strides(self.shape)

	def __str__(self):
		return array_print(self, ', ', lambda x: ' {0} '.format(x))

	def __getitem__(self, item):
		a_inqry = mdarray_inquery(self)
		new_shape, _gslice_list = make_gslice_list(item, a_inqry)

		tmp = _slice_array(self, _gslice_list)
		tmp = mdarray(shape=new_shape, data=tmp)

		return tmp


def arange(size):
	data = [i for i in range(size)]
	return mdarray(size=size, data=data)


def swap_axis(a, axis1, axis2):
	swap_item(a.strides)


shape = [3, 2, 5]
size = reduce(lambda x, y: x*y, shape)

a = arange(size)

a.reshape(shape)

print(mdarray_inquery(a))
print(a)

t = [i for i in range(size)]
t = np.asarray(t).reshape(shape)

# print(t)

# ix = [[0, 1, 2], nan, 0, nan]
# print(a[ix])

# print('\n')

# print(a[:, 0])
#
# strides = get_strides(shape[::-1])[::-1]
#
# slc = [[0, 1], NAN, [0]]
#
# iter_groups = make_iter_groups(slc, [])
# slcs = []
# for i in iter_groups:
# 	slcs += [gslice(i, strides, shape)]
#
# slicd_array = _slice_array(a.flatten(), slcs, [])
# print(slicd_array)

# print(slicd_array)

# strides = get_strides(slc.shape[::-1])[::-1]
# shape = slc.shape
# dim = slc.dim
#
#
# tt = ap(slicd_array, dim, shape, strides, [0]*dim, 0, sep='', formatter=lambda x: ' {0} '.format(x))
# print('\n')
# print(tt)
