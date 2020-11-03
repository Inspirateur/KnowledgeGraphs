from collections import defaultdict
from typing import Callable
import numpy as np
from tqdm import tqdm
from Models.KG import KG
from Structures.imap import IMap


class Proximity(KG):
	"""
	A baseline Knowledge Graph that ignores relation label
	and scores triplet via proximity of head and tail entities
	Similar to: https://arxiv.org/pdf/1501.03471.pdf, Ciampaglia et al. 2015
	"""
	emap: IMap
	distances: Callable
	r2t: dict

	def __init__(self, fct="shortest_path"):
		self.limit = 7
		self.fct = fct

	def load(self, train, valid, dataset: str):
		self.train(train, valid, dataset)

	def train(self, train, valid, dataset: str):
		self.r2t = defaultdict(set)
		for h, r, t in train:
			self.r2t[r].add(t)
		for r, tails in self.r2t.items():
			self.r2t[r] = np.array(list(tails))
		# TODO: rewrite with scipy
		"""
		comps = sorted(
			nx.connected_components(nx.Graph([(h, t) for h, _, t in train])),
			key=len, reverse=True
		)
		adjs = [nx.Graph() for _ in range(len(comps))]
		for h, _, t in train:
			_i = 0
			while h not in comps[_i]:
				_i += 1
			adjs[_i].add_edge(h, t)

		def shortest_path(source, targets):
			for adj in adjs:
				if source in adj:
					break
			else:
				return np.full(len(targets), fill_value=float("+inf"))
			dist = shortest_path_length(adj, source)
			dist = np.array([dist.get(_t, float("+inf")) for _t in targets], dtype=np.float)
			dist[dist <= 1] = float("+inf")
			return dist
	
		self.distances = shortest_path
		"""

	def link_completion(self, n, doubles):
		preds = []
		for h, r in tqdm(doubles, ncols=140, desc="Evaluating"):
			distances: np.ndarray = self.distances(h, self.r2t[r])
			_n = min(len(distances)-1, n)
			ind = distances.argpartition(_n)[:_n]
			tails: np.ndarray = self.r2t[r][ind]
			preds.append(tails.tolist())
		return preds
