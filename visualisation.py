from collections import defaultdict
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
from networkx.generators import ego_graph
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from datasets import load_dataset
from Structures.imap import IMap
from Structures.graph3d import Graph3D


def plot_entity(embeddings, emap: dict):
	writer = SummaryWriter()
	writer.add_embedding(embeddings, metadata=list(emap.keys()))
	writer.close()


def plot_neighbors(dataset: str):
	train, valid, test = load_dataset(dataset)
	counts = defaultdict(int)
	for h, r, t in train:
		counts[h] += 1
		counts[t] += 1
	x = np.array(list(counts.values()))
	plt.hist(x, bins=range(0, 150))
	plt.axvline(x.mean(), color="red", linestyle="dashed")
	plt.axvline(np.median(x), color="black")
	plt.title(f"Node degrees in {dataset}")
	plt.xlabel("degree")
	plt.ylabel("# of nodes")
	plt.show()


def plot_connected_comps(dataset: str):
	train, valid, test = load_dataset(dataset)
	height = np.array(
		[
			len(c)
			for c in sorted(
				nx.connected_components(nx.Graph([(h, t) for h, _, t in train])),
				key=len,
				reverse=True,
			)
		]
	)
	height = height / height.sum()
	plt.bar(list(range(1, len(height) + 1)), height)
	plt.annotate(f"{height[0]:.1%}", xy=(1, height[0] - 0.05), ha="center")
	plt.axhline(height.mean(), color="red", linestyle="dashed")
	plt.axhline(np.median(height), color="black")
	plt.title(f"Proportions of connected components in {dataset}")
	plt.show()


def plot_test_distances(dataset: str):
	limit = 10
	train, valid, test = load_dataset(dataset)
	# map entities to an id
	emap = IMap()
	for h, _, t in train:
		emap.put(h)
		emap.put(t)
	# build the kg
	kg = lil_matrix((len(emap), len(emap)), dtype=np.uint16)
	for h, _, t in train:
		kg[emap[h], emap[t]] = 1
	kg = kg.tocsr()
	test.sort(key=lambda hrt: hrt[0])
	distances = []
	_h = None
	shortest = None
	for h, _, t in tqdm(test, desc="Distances"):
		if _h != h:
			shortest = dijkstra(
				kg, limit=limit, indices=emap[h], return_predecessors=False
			)
			_h = h
		distances.append(shortest[emap[t]])
	distances = np.array(distances)
	distances[distances > limit] = limit + 1
	plt.hist(distances, bins=range(0, limit + 2))
	plt.axvline(distances.mean(), color="red", linestyle="dashed")
	plt.axvline(np.median(distances), color="black")
	plt.title(f"Distances of test triples in training graph in {dataset}")
	plt.xlabel("distance")
	plt.ylabel("# of nodes")
	plt.show()


def browse_dataset(dataset):
	train, valid, test = load_dataset(dataset)

	graph = Graph3D()
	graph.add(*train)
	graph.browse()


def known_candidates(dataset):
	train, valid, test = load_dataset(dataset)

	candidates = defaultdict(set)
	nodes = set()
	for h, r, t in train:
		nodes.add(h)
		nodes.add(t)
		candidates[r].add(t)

	def known_rt(data):
		known = 0
		for _h, _r, _t in data:
			if _t in candidates[_r]:
				known += 1
		return known/len(data)

	def known_e(data):
		known = 0
		for _h, _r, _t in data:
			if _h in nodes and _t in nodes:
				known += 1
		return known/len(data)

	print(f"Known <r, t> / Known e for {dataset}-valid: {known_rt(valid):.1%} / {known_e(valid):.1%}")
	print(f"Known <r, t> / Known e for {dataset}-test:  {known_rt(test):.1%} / {known_e(test):.1%}")


if __name__ == "__main__":
	from datasets import WIKI, WN18RR, FB15k

	known_candidates(FB15k)
