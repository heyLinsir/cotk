def new_module(name, doc=None):
	import sys
	from types import ModuleType
	m = ModuleType(name, doc)
	m.__file__ = name + '.py'
	sys.modules[name] = m
	return m

try:
	import pytorch_pretrained_bert
except ImportError as err:
	m = new_module("pytorch_pretrained_bert", "a dummy package from ._utils.imports")

	def make_attr(err):
		def __getattr__(_):
			raise e
	
	m.__getattr__ = make_attr(err)
