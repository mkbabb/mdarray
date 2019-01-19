import numpy as np
from functools import reduce
import math

from mdarray_helper import pair_wise_accumulate, pair_wise, swap_item, get_strides
from mdarray_indexing import (
	iter_axis, make_iter_list, gslice, _slice_array, make_array_indicies, mdarray_inquery,
	make_nested, flatten,
	)
from mdarray_formatting import array_print, pad_array_fmt
from mdarray_types import inf, nan


class mdarray(object):
	def __init__(self, shape=None, size=None, data=None):
		if not shape:
			if not size:
				raise ZeroDivisionError
			self.shape = [size]
			self.size = size
		else:
			self.shape = shape
			self.get_size()

		self.mdim = len(self.shape)
		self.strides = get_strides(self.shape)

		self.data = data if data else [0]*self.size

		self.dtype = type(self.data[0])

	def reshape(self, new_shape):
		new_size = reduce(lambda x, y: x*y, new_shape)

		if new_size != self.size:
			raise ZeroDivisionError

		self.shape = new_shape

		self._get_mdim()
		self._get_strides()
		self.size = new_size

	def T(self, axis1=0, axis2=0):
		if axis1 == axis2 == 0:
			axis1 = 1
			axis2 = 0

		self.strides = swap_item(self.strides, axis1, axis2)
		self.shape = swap_item(self.shape, axis1, axis2)

		return self

	def to_list(self):
		return make_nested(self.data)

	def astype(self, type):
		try:
			if type == complex:
				pass
			else:
				pass
		except TypeError:
			raise TypeError

	def _get_mdim(self):
		self.mdim = len(self.shape)

	def _get_size(self):
		self.size = reduce(lambda x, y: x*y, self.shape)

	def _get_strides(self):
		self.strides = get_strides(self.shape)

	def __str__(self):
		return array_print(self, ', ', lambda x: ' {0} '.format(x))

	def __setitem__(self, key, value):
		tmp = self[key]
		print(tmp)

	def __getitem__(self, item):
		gslice_list = make_array_indicies(a, item)
		print(gslice_list)


		# tmp = _slice_array(self, _gslice_list)
		# tmp = mdarray(shape=new_shape, data=tmp)

		return 0

	def __iter__(self):
		self.pos = 0
		return self

	def __next__(self):
		ppos = self.pos
		self.pos += 1

		if self.pos == self.size:
			raise StopIteration
		else:
			return self.data[ppos]

	def __len__(self):
		return self.size



def arange(size):
	data = [i for i in range(size)]
	return mdarray(size=size, data=data)


def swap_axis(a, axis1, axis2):
	swap_item(a.strides, axis1, axis2)


def tomdarray(a):
	if isinstance(a, mdarray):
		return a
	else:
		if isinstance(a, list):
			a, _, shape = flatten(a)
			md = mdarray(shape=shape, data=a)
			return md
		elif isinstance(a, dict):
			tmp = [[i, j] for i, j in a.items()]
			return tomdarray(tmp)


# lst = [[[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]],
#
#        [[10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]]]
#
# md = tomdarray(lst)
# for i in md:
# 	print(i)


# flt = flatten(lst)

shape = [2, 3, 3]
size = reduce(lambda x, y: x*y, shape)

a = arange(size)

a.reshape(shape)

print(a)


lst = make_iter_list([nan, 0, [0, 1]])
print(lst)
# ix1 = gslice([nan, 0, [0, 1, 2]], a).get_slice()
#
# print(ix1)

# too = make_array_indicies(a, ix1)
# print(too)



#
#
# v = _slice_array(a, too)
# print(v)



# t = a[ix1]
# print(t)


#
# t = make_nested(a)
# print(t)
#
# flt, dimc, shp = flatten(t)
# print(flt, dimc, shp)
# s = array_print(a, ' , ', pad_array_fmt(a))
# print(s)

# print(mdarray_inquery(a))
# print(a)
#
# t = [i for i in range(size)]
# t = np.asarray(t).reshape(shape)


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
