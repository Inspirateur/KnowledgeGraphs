from collections import defaultdict
import json
import numpy as np
from tqdm import trange
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from datasets import load_dataset, FB15k
from Structures.imap import IMap


def sparse_graph(data):
	emap = IMap()
	for h, r, t in data:
		emap.put(h)
		emap.put(t)
	g = lil_matrix((len(emap), len(emap)))
	for h, r, t in data:
		g[emap[h], emap[t]] = 1
	return g.tocsr()


def targets(data, dataset: str, min_dist=2, max_dist=3):
	try:
		with open(f"Structures/bad_ex_{dataset}.json", "r") as f:
			return json.load(f)
	except FileNotFoundError:
		emap = IMap()
		r_t = defaultdict(set)
		h_r = defaultdict(set)
		for h, r, t in data:
			emap.put(h)
			emap.put(t)
			r_t[r].add(emap[t])
			h_r[emap[h]].add(r)
		g = lil_matrix((len(emap), len(emap)))
		for h, r, t in data:
			g[emap[h], emap[t]] = 1
		g = g.tocsr()
		ts = []
		for i in trange(len(emap), desc="Bad examples", ncols=140):
			rel_inds = set()
			for r in h_r[i]:
				rel_inds |= r_t[r]
			dists = dijkstra(
				g, directed=False, unweighted=True, indices=i,
				return_predecessors=False, limit=max_dist
			)
			dists_inds = set(np.asarray((min_dist <= dists) & (dists <= max_dist)).nonzero()[0].tolist())
			ts.append(list(dists_inds ^ rel_inds))
		with open(f"Structures/bad_ex_{dataset}.json", "w") as f:
			json.dump(ts, f)
		return ts
