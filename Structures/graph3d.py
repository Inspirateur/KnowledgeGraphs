from collections import defaultdict
from Structures.imap import IMap


class Graph3D:
	def __init__(self):
		# a mapping between english names and entity number
		self.emap = IMap()
		# a mapping between english names and relation number
		self.rmap = IMap()
		# the Knowledge Graph
		self.kg = defaultdict(list)

	def __sizeof__(self):
		return self.emap.__sizeof__() + self.rmap.__sizeof__() + self.kg.__sizeof__() + \
			sum(edges.__sizeof__() for edges in self.kg.values())

	def __getitem__(self, item):
		if isinstance(item, int):
			return self.kg[item]
		return self.kg[self.emap[item]]

	def __iter__(self):
		return iter(self.emap.keys())

	def __len__(self):
		return len(self.emap)

	def __contains__(self, item):
		return item in self.emap and self.emap[item] in self.kg

	def relations(self, entity):
		return [(entity, r, t) for r, t in self[entity]]

	def add(self, *triplets):
		for h, r, t in triplets:
			self.emap.put(h)
			self.rmap.put(r)
			self.emap.put(t)
			self.kg[self.emap[h]].append((self.rmap[r], self.emap[t]))
			self.kg[self.emap[t]].append((-self.rmap[r], self.emap[h]))

	def inspect(self, entity):
		return (
			("-> "+self.rmap.rget(r), self.emap.rget(t))
			if r >= 0 else
			("<- "+self.rmap.rget(-r), self.emap.rget(t))
			for r, t in self.kg[self.emap[entity]]
		)

	def out(self, entity):
		for r, t in self[entity]:
			if r > 0:
				yield self.rmap.rget(r), self.emap.rget(t)

	def to_relfirst_dict(self):
		res = {}
		for h in self.emap.values():
			res[h] = defaultdict(list)
			for r, t in self[h]:
				res[h][r].append(t)
			res[h] = dict(res[h])
		return res
