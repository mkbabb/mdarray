from functools import reduce


def swap_item(a, ix1, ix2):
	t = a[ix1]
	a[ix1] = a[ix2]
	a[ix2] = t


def pair_wise(a1, a2, func):
	tmp = [0]*len(a1)
	for n, i in enumerate(a1):
		t = func(i, a2[n])
		tmp[n] = t
	return tmp


def pair_wise_accumulate(a1, a2):
	return reduce(lambda x, y: x + y, pair_wise(a1, a2, lambda x, y: x*y))
