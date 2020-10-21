from random import randint
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
