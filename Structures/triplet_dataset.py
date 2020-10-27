from collections import defaultdict
from math import exp
from random import choice, randint
import torch.utils.data as torchdata
from Structures.imap import IMap
from Structures.graph3d import Graph3D


class TripleData(torchdata.Dataset):
	def __init__(self, dataset, emap: IMap, rmap: IMap):
		self.dataset = dataset
		self.emap = emap
		self.rmap = rmap

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		h, r, t = self.dataset[item]
		pos_triple = [self.emap[h], self.rmap[r], self.emap[t]]
		if randint(0, 1):
			neg_triple = [randint(1, len(self.emap)-1), self.rmap[r], self.emap[t]]
		else:
			neg_triple = [self.emap[h], self.rmap[r], randint(1, len(self.emap) - 1)]
		return pos_triple, neg_triple


class TripleNData(torchdata.Dataset):
	def __init__(self, dataset, graph: Graph3D):
		self.dataset = dataset
		self.graph = graph

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		h, r, t = self.dataset[item]
		pos_triple = [self.graph.emap[h], self.graph.rmap[r], self.graph.emap[t]]
		if randint(0, 1):
			neg_triple = [randint(1, len(self.graph.emap) - 1), self.graph.rmap[r], self.graph.emap[t]]
		else:
			neg_triple = [self.graph.emap[h], self.graph.rmap[r], randint(1, len(self.graph.emap) - 1)]
		pos_neigh = self.graph.relations(pos_triple[0]) if pos_triple[0] != 0 else [(0, 0, 0)]
		neg_neigh = self.graph.relations(neg_triple[0]) if neg_triple[0] != 0 else [(0, 0, 0)]
		return pos_triple, neg_triple, pos_neigh, neg_neigh


class TriplePathsData(torchdata.Dataset):
	def __init__(self, dataset, graph: Graph3D, depth, ppr):
		self.dataset = dataset
		self.graph = graph
		self.depth = depth
		self.ppr = ppr

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		h, r, t = self.dataset[item]
		paths = defaultdict(lambda: defaultdict(float))
		# for every target and direct relation from h
		for r0, node0 in self.graph[h]:
			# we skip the connexion <h, r, t> we're training on
			if node0 == t:
				continue
			# walk ppr (paths per relations) depth-limited paths
			for i in range(self.ppr):
				visited = {h}
				path = [r0]
				node = t
				try:
					# walk a depth limited random path
					for d in range(self.depth - 1):
						if all(neigh in visited for _, neigh in self.graph[node]):
							raise ValueError("Walk is forced to cycle")
						_r, neigh = choice(self.graph[node])
						while neigh in visited:
							_r, neigh = choice(self.graph[node])
						path.append(_r)
						visited.add(node)
						node = neigh
						# break before the end if we reach the target
						if neigh == t:
							break
					# add 1 for the end node in the path distribution
					paths[tuple(path)][node] += 1
				except ValueError:
					pass
		# normalize path distriubtions with softmax
		for path in paths:
			total = sum(exp(count) for count in paths[path].values())
			for node in paths[path]:
				paths[path][node] = exp(paths[path][node]) / total
		# path score = softmax(prob of reaching t with path)
		path_list = []
		path_scores = []
		for path in paths:
			path_list.append(path)
			path_scores.append(paths[path][t])
		return r, path_list, path_scores
