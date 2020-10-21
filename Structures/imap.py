class IMap(dict):
	"""
	A convenience dict for mapping names to unique integers from offset to len(), its behavior differs in 2 ways:
	> <unk> is a special entry at index 0 that gets returned when an unknown entry is accessed
	> rget(i) is a shorthand for list(self.keys())[i]
	"""
	def __init__(self):
		dict.__init__(self)
		self._id2e = None
		self.put("<unk>")

	def __getitem__(self, item) -> int:
		try:
			return dict.__getitem__(self, item)
		except KeyError:
			return 0

	def put(self, item):
		if item not in self:
			self[item] = len(self)
			self._id2e = None

	def rget(self, i: int):
		try:
			return self._id2e[i]
		except TypeError:
			self._id2e = list(self.keys())
			return self._id2e[i]
