from functools import reduce
from mdarray_helper import pair_wise_accumulate
from mdarray_types import inf, nan
import math


def array_print(a, sep='', formatter=None):
	data = a.data
	mdim = a.mdim
	shape = a.shape
	strides = a.strides

	ix1 = [0]*mdim
	ix2 = 0

	if not formatter:
		formatter = lambda x: '{0}'.format(x)

	def recurse(ix1, ix2):
		axis = shape[ix2]
		remaining_axes = mdim - ix2

		s = ''
		if remaining_axes == 1:
			for i in range(axis):
				ix1[mdim - 1] = i
				ix3 = pair_wise_accumulate(ix1, strides)

				try:
					a_val = data[ix3]
				except IndexError:
					a_val = nan

				val = formatter(a_val)
				s += (val + sep) if i < axis - 1 else val
		else:
			new_line = '\n'*(remaining_axes - 1)
			for i in range(axis):
				ix1[ix2] = i

				val = recurse(ix1, ix2 + 1)

				s += ' '*(ix2 + 1) + val if i > 0 else val
				s += sep.strip() if i < axis - 1 else ''
				s += new_line if i != axis - 1 else ''

		s = '[{0}]'.format(s)
		return s

	return recurse(ix1, ix2)


def pad_array_fmt(a):
	max_len = len(str(max(a.data, key=lambda x: len(str(x)))))

	fmmter = lambda x: '{0}{1}{2}'.format(' '*math.ceil((max_len - len(str(x)))/2), x,
	                                      ' '*math.floor((max_len - len(str(x)))/2))
	return fmmter
