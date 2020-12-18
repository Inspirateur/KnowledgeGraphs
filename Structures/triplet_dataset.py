from collections import defaultdict
import functools
from math import exp, ceil
from random import sample, randint
import torch
import torch.utils.data as torchdata
from torch.nn.utils.rnn import pad_sequence
import unidecode
from scipy.sparse.csgraph import dijkstra
from Structures.imap import IMap
from Structures.graph3d import Graph3D
from Structures.graph import targets


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
	def __init__(self, dataset, graph: Graph3D, depth):
		self.dataset = dataset
		self.graph = graph
		self.depth = depth

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		h, r, t = self.dataset[item]
		paths = self.graph.random_walks_r(h, r, max_depth=self.depth)
		# path score = prob of reaching t with path
		path_list = []
		path_scores = []
		for path, score in paths.items():
			path_list.append(path)
			path_scores.append(paths[path][self.graph.emap[t]])
		return self.graph.emap[h], self.graph.rmap[r], path_list, path_scores


class TriplePathCtxData(torchdata.IterableDataset):
	def __init__(self, data, graph: Graph3D, depth, bad_exs, cmap, neg_per_pos):
		assert neg_per_pos > 0
		self.dataset = data
		self.graph = graph
		self.depth = depth
		self.neg_per_pos = neg_per_pos
		self.bad_exs = bad_exs
		self.cmap = cmap

	def __len__(self):
		return len(self.dataset)*(self.neg_per_pos+1)

	def _encode(self, txt):
		if not txt:
			return torch.tensor([0])
		return torch.tensor([
			self.cmap[c]
			for c in unidecode.unidecode(txt.replace('_', ' ').lower())
		])

	def _encode_paths(self, paths):
		if not paths:
			return torch.zeros((1, 1, self.depth), dtype=torch.long)
		return pad_sequence([
			pad_sequence([self._encode(r) for r in path])
			for path in paths
		])

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:
			s = slice(len(self.dataset))
		else:
			per_worker = len(self.dataset)/worker_info.num_workers
			it_start = int(worker_info.id*per_worker)
			it_end = int((worker_info.id+1)*per_worker)
			s = slice(it_start, it_end)
		for h, r, t in self.dataset[s]:
			bad_exs = sample(self.bad_exs[self.graph.emap[h]], self.neg_per_pos)
			paths = self.graph.random_paths(h, bad_exs+[t], max_depth=self.depth)
			h = self._encode(h)
			r = self._encode(r)
			yield h, r, self._encode(t), self._encode_paths(paths[t]), 1
			for bad_t in bad_exs:
				yield h, r, self._encode(self.graph.emap.rget(bad_t)), self._encode_paths(paths[bad_t]), 0
