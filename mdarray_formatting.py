from functools import reduce
from mdarray_helper import pair_wise_accumulate


def array_print(a, sep='', formatter=None):
	data = a.data
	dim = a.dim
	shape = a.shape
	strides = a.strides

	ix1 = [0]*dim
	ix2 = 0

	if not formatter:
		formatter = lambda x: '{0}'.format(x)

	def recurse(ix1, ix2):
		axis = shape[ix2]
		remaining_axes = dim - ix2

		s = ''
		if remaining_axes == 1:
			for i in range(axis):
				ix1[dim - 1] = i
				ix3 = pair_wise_accumulate(ix1, strides)

				a_val = data[ix3]

				val = formatter(a_val)
				s += (val + sep) if i < axis - 1 else val
		else:
			new_line = '\n'*(remaining_axes - 1)
			for i in range(axis):
				ix1[ix2] = i

				val = recurse(ix1, ix2 + 1)

				s += ' '*(ix2 + 1) + val if i > 0 else val
				s += new_line if i != axis - 1 else ''

		s = '[{0}]'.format(s)

		return s

	return recurse(ix1, ix2)
