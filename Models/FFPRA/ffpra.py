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
class FFPRAModule(nn.Module):
	def __init__(self, num_ents, num_rels, embed_size, depth):
		super().__init__()
		self.f1 = nn.Linear(embed_size*(depth+2), embed_size*2)
		self.f2 = nn.Linear(embed_size*2, embed_size)
		self.f3 = nn.Linear(embed_size, 10)
		self.f4 = nn.Linear(10, 1)
		self.r_embed = nn.Embedding(num_rels, embed_size)
		self.e_embed = nn.Embedding(num_ents, embed_size)

	def embed_rel(self, rel):
		"""
		:param rel: [batch]
		:return: [embed size, batch]
		"""
		return (torch.sign(rel)[:, None]*self.r_embed(torch.abs(rel))).transpose(0, 1)

	def embed_paths(self, paths: torch.Tensor):
		"""
		:param paths: [n, depth, batch]
		:return: [n, depth, embed size, batch]
		"""
		n, depth, batch_size = paths.shape
		return self.embed_rel(paths.view(-1)).reshape(n, depth, -1, batch_size)

	def forward(self, h, r, paths):
		"""
		:param h: [batch]
		:param r: [batch]
		:param paths: [n, depth, batch]
		:return: scores: [n, batch]
		"""
		# [n, depth, embed size, batch]
		paths = self.embed_paths(paths)
		n, depth, embed_size, batch_size = paths.shape
		# [depth, embed size, n*batch]
		paths = paths.reshape(depth, embed_size, n*batch_size)
		# [embed size, batch]
		h = self.e_embed(h).transpose(0, 1)
		# [embed size, batch]
		r = self.embed_rel(r)
		# [1, embed size, n*batch]
		r = r.expand(n, embed_size, batch_size).reshape(1, -1, n*batch_size)
		# [1, embed size, n*batch]
		h = h.expand(n, embed_size, batch_size).reshape(1, -1, n*batch_size)
		# [n*batch, embed size*(depth+1)]
		in1 = torch.cat([h, r, paths], dim=0).view(n*batch_size, -1)
		# [n*batch, embed_size*2]
		out1 = torch.relu(self.f1(in1))
		# [n*batch, embed_size]
		out2 = torch.relu(self.f2(out1))
		# [n*batch, 10]
		out3 = torch.relu(self.f3(out2))
		# [n*batch, 1]
		out4 = self.f4(out3)
		# [n, batch size]
		return out4.view(n, batch_size)


class FFPRA(KG):
	"""
	Same as PRA but using Feed Forward NN
	"""
	module: FFPRAModule
	optimizer: optim
	graph: Graph3D

	def __init__(self, depth=3, embed_size=50, lr=0.01):
		self.path = "Models/FFPRA/save.pt"
		if torch.cuda.is_available():
			print("Using the GPU")
			self.device = torch.device("cuda")
		else:
			print("Using the CPU")
			self.device = torch.device("cpu")
		self.depth = max(2, depth)
		self.embed_size = embed_size
		self.lr = lr
		self.batch_size = 4
		self.loss = nn.MSELoss()

	def epoch(self, it, train=True):
		roll_loss = deque(maxlen=50 if train else None)
		for h, r, paths, scores in it:
			if paths:
				h = h.to(self.device)
				r = r.to(self.device)
				paths = torch.stack(
					[torch.stack(path) for path in paths]
				).to(torch.long).to(self.device)
				scores = torch.stack(scores).to(torch.float).to(self.device)

				self.optimizer.zero_grad()
				# feed the head and the relation
				preds = self.module(h, r, paths)
				loss = self.loss(preds, scores)
				# learn
				if train:
					loss.backward()
					self.optimizer.step()
				roll_loss.append(loss.item())
				# display loss
				it.set_postfix_str(
					f"{'' if train else 'val '}loss: {sum(roll_loss)/len(roll_loss):.2f}"
				)
		return sum(roll_loss)/len(roll_loss) if roll_loss else None

	def train(self, train, valid, dataset: str):
		path = "Models/DLPRA/save.pt"
		# construct a knowledge graph from training triplets
		self.graph = Graph3D()
		self.graph.add(*train)

		train_batch = data.DataLoader(
			TriplePathsData(train, self.graph, self.depth), batch_size=self.batch_size
		)
		valid_batch = data.DataLoader(
			TriplePathsData(valid, self.graph, self.depth), batch_size=self.batch_size
		)

		# prepare the model
		self.module = FFPRAModule(
			len(self.graph.emap), len(self.graph.rmap), embed_size=self.embed_size, depth=self.depth
		).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		# train it
		epoch = 1
		best_val = float("+inf")
		patience = 3
		p = patience
		print(f"Early stopping with patience {patience}")
		while epoch <= 3:
			# while p > 0:
			print(f"Epoch {epoch} (p {p})")

			# training
			self.module.train()
			train_it = tqdm(train_batch, desc="\tTraining", file=sys.stdout, ncols=140)
			self.epoch(train_it)

			# validation
			self.module.eval()
			valid_it = tqdm(valid_batch, desc="\tValidating", file=sys.stdout, ncols=140)
			with torch.no_grad():
				v_loss = self.epoch(valid_it, train=False)
			if v_loss and v_loss < best_val:
				torch.save(self.module, path)
				best_val = v_loss
				p = patience
			else:
				p -= 1
			epoch += 1
			print()
		torch.save(self.module, path)

	def load(self, train, valid, dataset: str):
		self.graph = Graph3D()
		self.graph.add(*train)
		self.module = FFPRAModule(
			len(self.graph.emap), len(self.graph.rmap), embed_size=self.embed_size, depth=self.depth
		).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		self.module = torch.load(self.path)

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
				paths = self.graph.random_walks_r(h, r, self.depth)
				if not paths:
					candidates = [self.graph.emap.rget(c) for c in self.graph.r_t[self.graph.rmap[r]]]
					if candidates:
						preds.append(candidates[:n])
					else:
						preds.append([choice(self.graph) for _ in range(n)])
					continue
				# score the paths with the model
				path_list = torch.stack([torch.tensor(path) for path in paths]).unsqueeze(2).to(self.device)
				h_embed = torch.tensor([self.graph.emap[h]]).to(self.device)
				r_embed = torch.tensor([self.graph.rmap[r]]).to(self.device)
				scores = self.module(h_embed, r_embed, path_list)
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
