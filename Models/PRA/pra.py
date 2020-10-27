from collections import defaultdict, Counter, deque
import pickle
from random import choice
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from Models.KG import KG
from Structures.graph3d import Graph3D
from Structures.imap import IMap


def add_top_n(dists: dict, candidates, n, banned):
	pred = deque(maxlen=n)
	pred_d = deque(maxlen=n)
	for t in candidates:
		if t not in banned and (not len(pred) or dists[t] <= pred_d[-1]):
			pred.append(t)
			pred_d.append(dists[t])
		if len(pred) == pred.maxlen and pred_d[0] == 2:
			break
	return set(pred)


class PRA(KG):
	# <relation, <path, score>>
	paths: Dict[int, Dict[Tuple, float]]
	# <entity <relation, [entities]>>
	graph: Dict[int, Dict[int, List]]
	# <relation, tail>
	r2t: dict
	# id <=> entity
	emap: IMap
	# id <=> relations
	rmap: IMap

	def __init__(self, depth=3, ppr=50):
		self.depth = max(2, depth)
		self.ppr = ppr

	def train(self, train, valid, dataset: str):
		graph = Graph3D()
		graph.add(*train)
		self.paths = defaultdict(lambda: defaultdict(float))
		totals = defaultdict(lambda: 0)
		# for every node in the graph
		for h in tqdm(graph, ncols=140, desc="Finding paths"):
			# define the list of targets
			targets = {t: r for r, t in graph[h]}
			# for every target and direct relation
			for r, t in graph[h]:
				# walk ppr (paths per relations) depth-limited paths
				for i in range(self.ppr):
					visited = {h}
					path = [r]
					node = t
					try:
						for d in range(self.depth-1):
							if all(neigh in visited for _, neigh in graph[node]):
								raise ValueError("Walk is forced to cycle")
							_r, neigh = choice(graph[node])
							while neigh in visited:
								_r, neigh = choice(graph[node])
							path.append(_r)
							if neigh != t and neigh in targets:
								path = tuple(path)
								self.paths[targets[neigh]][path] += 1
								totals[path] += 1
								break
							visited.add(node)
							node = neigh
						else:
							path = tuple(path)
							totals[path] += 1
					except ValueError:
						pass
		# normalize path counts with total and turn defaultdicts into dict
		for r in self.paths:
			self.paths[r] = dict(self.paths[r])
			for path in self.paths[r]:
				self.paths[r][path] /= totals[path]
		self.paths = dict(self.paths)
		# pickle paths
		with open(f"Models/PRA/paths_{dataset}.pkl", "wb") as f:
			pickle.dump(self.paths, f)
		# save emap and rmap
		self.emap = graph.emap
		self.rmap = graph.rmap
		# re-organize the graph into a faster structure for prediction time
		self.graph = graph.to_relfirst_dict()

	def load(self, train, valid, dataset: str):
		with open(f"Models/PRA/paths_{dataset}.pkl", "rb") as f:
			self.paths = pickle.load(f)
		graph = Graph3D()
		graph.add(*train)
		self.emap = graph.emap
		self.rmap = graph.rmap
		self.graph = graph.to_relfirst_dict()
		self.r2t = defaultdict(set)
		for h, r, t in train:
			self.r2t[r].add(t)

	@staticmethod
	def inspect_paths(dataset: str):
		import matplotlib.pyplot as plt
		from datasets import load_dataset

		with open(f"Models/PRA/paths_{dataset}.json", "r") as f:
			paths = pickle.load(f)
		train, valid, test = load_dataset(dataset)
		rels = set()
		for h, r, t in train:
			rels.add(r)
		x = np.array(list(len(p) for p in paths.values()))
		plt.hist(x, bins=range(0, np.amax(x)+1, 25))
		plt.axvline(x.mean(), color="red", linestyle="dashed")
		plt.axvline(np.median(x), color="black")
		plt.title(f"Alternative len 3- paths in {dataset}")
		plt.xlabel("# paths")
		plt.ylabel("# relations")
		plt.show()

	def reach(self, h, path):
		targets = Counter()
		for i in range(50):
			node = h
			for r in path:
				if r not in self.graph[node]:
					break
				node = choice(self.graph[node][r])
			else:
				targets[node] += 1
		return targets

	def link_completion(self, n, doubles):
		preds = []
		for h, r in tqdm(doubles, desc="Evaluating", ncols=140):
			pred = Counter()
			h = self.emap[h]
			r = self.rmap[r]
			# use the alternative paths to make predictions
			if h in self.graph:
				paths = self.paths.get(r, dict())
				for path, weight in paths.items():
					targets = self.reach(h, path)
					total = sum(targets.values())
					for t, w in targets.items():
						pred[t] += weight*w/total
			preds.append([self.emap.rget(p) for p, _ in pred.most_common(n)])
		return preds
