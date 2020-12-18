from collections import defaultdict
from random import random, choice
import readline
from Structures.imap import IMap


def depth_amount(max_depth, neigh_size):
	"""
	Returns a list of tuple: depth, amount of path>=5
	Make it so that depth u_n+1 is neigh_size times depth u_n,
	using a geometric serie formula. Because as path depth grows,
	exponentially more paths are needed to cover a fraction of possible paths.
	"""
	n = min(neigh_size**max_depth, 10_000)
	u = (1-neigh_size)/(1-neigh_size**(max_depth-1))
	for d in range(2, max_depth+1):
		yield d, max(int(u*n), 5)
		u *= neigh_size


class Graph3D:
	def __init__(self):
		# a mapping between english names and entity number
		self.emap = IMap()
		# a mapping between english names and relation number
		self.rmap = IMap()
		# all known r -> t
		self.r_t = defaultdict(set)
		# the Knowledge Graph
		self.kg = defaultdict(list)

	def __sizeof__(self):
		return (
			self.emap.__sizeof__()
			+ self.rmap.__sizeof__()
			+ self.kg.__sizeof__()
			+ sum(edges.__sizeof__() for edges in self.kg.values())
		)

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
			self.r_t[self.rmap[r]].add(self.emap[t])
			self.kg[self.emap[h]].append((self.rmap[r], self.emap[t]))
			self.kg[self.emap[t]].append((-self.rmap[r], self.emap[h]))

	def inspect(self, entity):
		return (
			("-> " + self.rmap.rget(r), self.emap.rget(t))
			if r >= 0
			else ("<- " + self.rmap.rget(-r), self.emap.rget(t))
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

	def random_walks_r(self, h, r, max_depth, neigh_ratio=0.5):
		"""
		Random walks from h that only ends in known t for r
		"""
		assert max_depth >= 2
		# <path, <end_node, count>>
		paths = defaultdict(lambda: defaultdict(float))
		candidates = self.r_t[self.rmap[r]]
		# if the node is unknown, return empty paths
		if h not in self or not candidates:
			return paths
		# dynamically choose a value for n
		neigh_size = max(int(len(self[h])*neigh_ratio), 5)
		# for each depth, amount of path
		for depth, amount in depth_amount(max_depth, neigh_size):
			# walk a random path
			for i in range(amount):
				path = []
				node = h
				# walk a random path with random depth (up to depth)
				for d in range(depth):
					_r, neigh = choice(self[node])
					path.append(_r)
					node = neigh
				if node in candidates:
					# pad every paths with 0 (for batching purposes)
					path += [0] * (max_depth - len(path))
					assert len(path) == max_depth
					# add 1 for the end node in the path distribution
					paths[tuple(path)][node] += 1

		# normalize path distributions
		for path in paths:
			total = sum(paths[path].values())
			for node in paths[path]:
				paths[path][node] = paths[path][node] / total
		return paths

	def random_paths(self, h, targets, max_depth, neigh_ratio=0.5):
		assert max_depth >= 2
		# {targets: paths}
		paths = defaultdict(set)
		# if the node is unknown, return empty paths
		if h not in self:
			return paths
		h = self.emap[h]
		# accelerate the search for paths to t
		neighs = {
			n: (-r, t)
			for t in targets
			for r, n in self[t]
		}
		# dynamically choose a value for n
		neigh_size = max(int(len(self[h])*neigh_ratio), 5)
		# for each depth, amount of path
		for depth, amount in depth_amount(max_depth, neigh_size):
			# walk a random path
			for i in range(amount):
				path = []
				node = h
				# walk a random path up to depth-1
				for d in range(depth-1):
					_r, neigh = choice(self[node])
					path.append(self.rmap.rget(_r))
					node = neigh
				if node in neighs:
					_r, _t = neighs[node]
					path.append(self.rmap.rget(_r))
					# pad every paths with 0 (for batching purposes)
					path += [''] * (max_depth - len(path))
					assert len(path) == max_depth
					# add the path
					paths[_t].add(tuple(path))
		return defaultdict(list, {t: list(paths) for t, paths in paths.items()})

	def browse(self):
		node = list(self.emap.keys())[1]

		def completer(text, state):
			options = [
				self.emap.rget(_t)
				for _, _t in self[node]
				if self.emap.rget(_t).startswith(text)
			]
			return options[state] if state < len(options) else None

		readline.parse_and_bind("tab: complete")
		readline.set_completer(completer)

		print(f"{node} |{len(self[node])}|:")
		for r, t in self.inspect(node):
			print(r, t)

		while True:
			print("Type stop to stop or or <entity> to explore the entity")
			inpt = input("")
			print()
			if inpt.lower() in {"stop", "quit", "exit"}:
				break
			if inpt in self:
				node = inpt
				print(f"{node} |{len(self[node])}|:")
				for r, t in self.inspect(node):
					print(r, t)
			else:
				print(f"{inpt} is not known in the knowledge graph")
