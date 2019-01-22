class mdarray_inquery(object):
	def __init__(self, *args, **kwargs):
		a = None
		self.mdim = 0
		self.shape = [0]
		self.size = 0
		self.strides = [0]
		self.dtype = None

		if len(args) == 1:
			a = args[0]
		else:
			try:
				a = kwargs.pop("array")
			except KeyError:
				for i, j in kwargs.items():
					setattr(self, i, j)
		if a:
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

		for i, j in self.__dict__.items():
			current_len = len(str(i))
			space = ' '*(max_len - current_len)
			s += '{0}:{1}{2}\n'.format(i, space, j)
		return s


class md_nan(object):
	def __init__(self):
		self.value = 'not a number!'

	def __eq__(self, other):
		if isinstance(other, md_nan):
			return True
		return False

	def __repr__(self):
		return 'nan'


class md_infp(object):
	def __init__(self):
		self.value = 'infinity!'

	def __eq__(self, other):
		return self

	def __gt__(self, other):
		return True

	def __lt__(self, other):
		return False

	def __mul__(self, other):
		return inf

	def __int__(self):
		return self

	def __float__(self):
		return self

	def __repr__(self):
		return 'inf'


class md_infn(object):
	def __init__(self):
		self.value = '-infinity!'

	def __eq__(self, other):
		return self

	def __gt__(self, other):
		return False

	def __lt__(self, other):
		return True

	def __mul__(self, other):
		return inf

	def __int__(self):
		return self

	def __float__(self):
		return self

	def __repr__(self):
		return '-inf'


_infp = md_infp()
_infn = md_infn()


class md_inf(md_infp):
	def __init__(self):
		super(md_inf, self).__init__()

	def __eq__(self, other):
		if isinstance(other, md_inf):
			return True
		return False

	def __gt__(self, other):
		if isinstance(other, md_inf):
			return False
		return True

	def __lt__(self, other):
		if isinstance(other, md_inf):
			return False
		return False

	def __mul__(self, other):
		if other < 0:
			return _infn
		else:
			return _infp

	def __repr__(self):
		return 'inf'


nan = md_nan()
inf = md_inf()
