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


infp = md_infp()
infn = md_infn()


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
			return infn
		else:
			return infp

	def __repr__(self):
		return 'inf'


nan = md_nan()
inf = md_inf()
