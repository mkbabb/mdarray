def get_strides(shape):
	strides = []
	tmp = 1
	for i in shape:
		strides += [tmp]
		tmp *= i

	return strides


def iter_array(shape, strides):
	N = len(shape)
	counters = [0]*N

	add = 0
	ix = 0

	while (counters[0] < shape[0]):
		counters[N - 1] += 1

		if (counters[N - 1] == shape[N - 1]):
			add = 0

			for n in range(N - 1, -1, -1):
				if counters[n] == shape[n] and n != 0:
					counters[n - 1] += 1
					counters[n] = 0

				add += counters[n]*strides[n]
			ix = add

		else:
			ix += strides[N - 1]
