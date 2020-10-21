from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
import networkx as nx
from datasets import load_dataset
from Structures.imap import IMap


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
	height = np.array([
		len(c) for c in sorted(
			nx.connected_components(nx.Graph([(h, t) for h, _, t in train])),
			key=len, reverse=True
		)
	])
	height = height/height.sum()
	plt.bar(list(range(1, len(height)+1)), height)
	plt.annotate(f"{height[0]:.1%}", xy=(1, height[0]-.05), ha="center")
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
			shortest = dijkstra(kg, limit=limit, indices=emap[h], return_predecessors=False)
			_h = h
		distances.append(shortest[emap[t]])
	distances = np.array(distances)
	distances[distances > limit] = limit+1
	plt.hist(distances, bins=range(0, limit+2))
	plt.axvline(distances.mean(), color="red", linestyle="dashed")
	plt.axvline(np.median(distances), color="black")
	plt.title(f"Distances of test triples in training graph in {dataset}")
	plt.xlabel("distance")
	plt.ylabel("# of nodes")
	plt.show()


if __name__ == '__main__':
	from datasets import WIKI, WN18RR, FB15k

	plot_test_distances(FB15k)
