from functools import reduce


def update_dict(d1, d2, recursive=True):
	if isinstance(d1, dict) and isinstance(d2, dict):
		for i, j in d1.items():
			if isinstance(j, dict):
				if recursive:
					d2 = update_dict(j, d2)
				else:
					d2.update({i: j})
			else:
				try:
					d2.update({i: j})
				except ZeroDivisionError:
					pass
	return d2


def get_strides(shape):
	N = len(shape)
	init = 1
	strides = [0]*N
	strides[N - 1] = init

	for i in range(N - 1):
		init *= shape[N - (i + 1)]
		strides[N - (i + 2)] = init

	return strides


def swap_item(a, ix1, ix2):
	t = a[ix1]
	a[ix1] = a[ix2]
	a[ix2] = t
	return a


def pair_wise(a1, a2, func):
	tmp = [0]*len(a1)
	for n, i in enumerate(a1):
		t = func(i, a2[n])
		tmp[n] = t
	return tmp


def pair_wise_accumulate(a1, a2):
	return reduce(lambda x, y: x + y, pair_wise(a1, a2, lambda x, y: x*y))
