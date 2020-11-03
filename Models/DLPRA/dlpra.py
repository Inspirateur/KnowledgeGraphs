from collections import deque, defaultdict
from random import choice
from typing import List
import sys
import torch
from torch.utils import data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Models.KG import KG
from Structures.graph3d import Graph3D
from Structures.triplet_dataset import TriplePathsData


# see https://github.com/python/mypy/issues/8795
# noinspection PyAbstractClass
class APRA(nn.Module):
	def __init__(self, num_rels, embed_size):
		super().__init__()
		self.path_encoder = nn.GRU(embed_size, embed_size, num_layers=2)
		self.attn = nn.MultiheadAttention(embed_size, 2)
		self.r_embed = nn.Embedding(num_rels, embed_size)

	def embed_rel(self, rel):
		return torch.sign(rel)[:, None]*self.r_embed(torch.abs(rel))

	def embed_paths(self, paths: torch.Tensor):
		shape = paths.shape
		paths = self.embed_rel(paths.view(-1))
		# path encoder expects [seq len, batch size, embed size]
		output, hn = self.path_encoder(paths.view(shape[1], shape[0]*shape[2], -1))
		return hn[-1].view(shape[0], shape[2], -1)

	def forward(self, r, paths):
		"""
		:param r: [1, batch]
		:param paths: [n, path size, batch]
		"""
		r = self.embed_rel(r).unsqueeze(0)
		paths = self.embed_paths(paths)
		attn_output, attn_scores = self.attn(r, paths, paths)
		return attn_scores


class DLPRA(KG):
	"""
	Same as PRA but using deep learning, more formally:
	Given r and a distribution of targets for each path predict t,
	using r as a query, and all paths as context we can use attention to give weight to paths on the fly
	"""
	module: APRA
	optimizer: optim
	graph: Graph3D

	def __init__(self, depth=3, walks=50, embed_size=50, lr=0.01):
		if torch.cuda.is_available():
			print("Using the GPU")
			self.device = torch.device("cuda")
		else:
			print("Using the CPU")
			self.device = torch.device("cpu")
		self.depth = max(2, depth)
		self.walks = walks
		self.embed_size = embed_size
		self.lr = lr
		self.batch_size = 64

	def criterion(self, scores, preds):
		# Basically a MSE (L2) Loss
		return torch.pow(scores-preds.view(scores.shape), 2).mean()

	def epoch(self, it, learn=True):
		roll_loss = deque(maxlen=50 if learn else None)
		for r, paths, scores in it:
			r = r.to(self.device)
			if paths:
				paths = torch.stack(
					[torch.stack(path) for path in paths]
				).to(torch.long).to(self.device)
				scores = torch.stack(scores).to(torch.long).to(self.device)

				self.optimizer.zero_grad()
				# feed the head and the relation
				preds = self.module(r, paths)
				loss = self.criterion(scores, preds)
				# learn
				if learn:
					loss.backward()
					self.optimizer.step()
				roll_loss.append(loss.item())
				# display loss
				it.set_postfix_str(
					f"{'' if learn else 'val '}loss: {sum(roll_loss)/len(roll_loss):.2f}"
				)
		return sum(roll_loss)/len(roll_loss)

	def train(self, train, valid, dataset: str):
		path = "Models/DLPRA/save.pt"
		# construct a knowledge graph from training triplets
		self.graph = Graph3D()
		self.graph.add(*train)
		train_batch = data.DataLoader(
			TriplePathsData(train, self.graph, self.depth, self.walks), batch_size=self.batch_size
		)
		valid_batch = data.DataLoader(
			TriplePathsData(valid, self.graph, self.depth, self.walks), batch_size=self.batch_size
		)

		# prepare the model
		self.module = APRA(len(self.graph.rmap), embed_size=self.embed_size).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		# train it
		epoch = 1
		best_val = float("+inf")
		patience = 5
		p = patience
		print(f"Early stopping with patience {patience}")
		# while p > 0:
		while epoch == 1:
			print(f"Epoch {epoch}")

			# training
			self.module.train()
			train_it = tqdm(train_batch, desc="\tTraining", file=sys.stdout, ncols=140)
			self.epoch(train_it)

			# validation
			self.module.eval()
			valid_it = tqdm(valid_batch, desc="\tValidating", file=sys.stdout, ncols=140)
			with torch.no_grad():
				v_loss = self.epoch(valid_it, learn=False)
			if v_loss < best_val:
				torch.save(self.module, path)
				best_val = v_loss
				p = patience
			else:
				p -= 1
			epoch += 1
			print()

	def load(self, train, valid, dataset: str):
		self.graph = Graph3D()
		self.graph.add(*train)
		self.module = APRA(len(self.graph.rmap), embed_size=self.embed_size).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)

	def link_completion(self, n, couples) -> List[str]:
		preds = []
		idx2e = list(self.graph.emap.keys())
		self.module.eval()
		with torch.no_grad():
			for h, r in tqdm(couples, desc="Evaluating", ncols=140):
				# if h is not known, return random candidates
				if h not in self.graph:
					preds.append([choice(self.graph) for _ in range(n)])
					continue
				# do random walks and collect the end nodes in a distribution
				paths = self.graph.random_walks(h, self.depth, self.walks)
				# score the paths with the model
				path_list = torch.stack([torch.tensor(path) for path in paths]).unsqueeze(2).to(self.device)
				r_embed = torch.tensor([self.graph.rmap[r]]).to(self.device)
				scores = self.module(r_embed, path_list).view(-1).tolist()
				# score the candidates
				candidates = defaultdict(float)
				for path, score in zip(paths.keys(), scores):
					for target, weight in paths[path].items():
						candidates[target] += score*weight
				# rank them
				preds.append([
					idx2e[node] for node, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
				][:n])
		return preds
