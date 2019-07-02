class dummy(object):
	def __init__(self, err):
		self.err = err

	def __getattr__(self, _):
		raise self.err
