class IMap(dict):
	"""
	A convenience dict for mapping names to unique integers from offset to len(), its behavior differs in 2 ways:
	> <unk> is a special entry at index 0 that gets returned when an unknown entry is accessed
	> rget(i) is a shorthand for list(self.keys())[i]
	"""
	def __init__(self, items=None):
		dict.__init__(self)
		self._id2e = None
		self.put("<unk>")
		if items:
			self.puts(items)

	def __getitem__(self, item) -> int:
		return dict.get(self, item, 0)

	def puts(self, items):
		for item in items:
			self.put(item)

	def put(self, item):
		if item not in self:
			self[item] = len(self)
			self._id2e = None

	def rget(self, i: int):
		try:
			return ("-" if i < 0 else "") + self._id2e[abs(i)]
		except TypeError:
			self._id2e = list(self.keys())
			return ("-" if i < 0 else "") + self._id2e[abs(i)]
