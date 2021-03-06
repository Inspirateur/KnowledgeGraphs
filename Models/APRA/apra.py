from collections import deque, defaultdict
import copy
import functools
import os.path
from random import choice
from typing import List
import sys
from linformer_pytorch import Linformer, Padder
import torch
from torch.utils import data as data
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Models.KG import KG
from Structures.graph3d import Graph3D
from Structures.triplet_dataset import TriplePathCtxData
from Structures.graph import targets
from Structures.imap import IMap
import Models.APRA.apra_visualize as plot


class CharLevelEncoder(nn.Module):
	def __init__(self, cvoc, embed_size, device):
		super().__init__()
		self.device = device
		self.embed_size = embed_size
		self.channels = 8
		self.cvoc = cvoc
		self.embedding = nn.Embedding(len(cvoc), self.channels)
		self.convs = nn.Sequential(
			nn.Conv1d(8, 16, kernel_size=5, padding=2),
			nn.LeakyReLU(),
			nn.MaxPool1d(3, padding=1),
			nn.Conv1d(16, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.MaxPool1d(3, padding=1),
			nn.Conv1d(32, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.MaxPool1d(3, padding=1),
			nn.Conv1d(64, 128, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.AdaptiveAvgPool1d(1),
		)
		self.ff = nn.Linear(128, embed_size)

	def forward(self, seq):
		"""
		Embed the sequence with character level embedding
		:param seq: [l, batch]
		:return: embed: [batch, embed_size]
		"""
		# [batch, channels, l]
		seq = self.embedding(seq.reshape(-1)).reshape(seq.shape[1], -1, seq.shape[0])
		out = self.convs(seq).squeeze()
		out = self.ff(out)
		return out


class RelationEncoder(nn.Module):
	def __init__(self, embed_size, num_layers=2):
		super().__init__()
		self.path_encoder = nn.GRU(embed_size, embed_size, 2)
		# FIXME: computing a new representation of r and every paths for each layer is very expensive (attention bottleneck)
		#  need to use a cheaper alternative
		self.layers = Padder(Linformer(
			256, embed_size, dim_d=None, dim_k=embed_size*2, dim_ff=embed_size*2, nhead=4, depth=num_layers
		))

	def forward(self, r, paths):
		"""
		Embed r using attention with paths as context
		:param r: [batch, embed size]
		:param paths: [n, depth, batch, embed size]
		:return: [batch, embed size] a new embedding of r
		"""
		n, depth, batch, embed_size = paths.shape
		# [depth, n*batch, embed size]
		paths = paths.reshape(depth, n*batch, embed_size)
		# [depth, n*batch, embed size]
		paths, _ = self.path_encoder(paths)
		# take the output of the GRU at the last timestep
		paths = paths[-1].reshape(n, batch, embed_size)
		# concatenate r and the encoded paths
		# [batch, n+1, embed size]
		src = torch.cat([r.unsqueeze(0), paths], dim=0).transpose(0, 1)
		# pass it through a transformer-like model and retrieve only the representation of r
		# [batch, n+1, embed size]
		src_encoded = self.layers(src)
		# [batch, embed size]
		return src_encoded[:, 0, :]


# see https://github.com/python/mypy/issues/8795
# noinspection PyAbstractClass
class APRAModule(nn.Module):
	def __init__(self, cvoc, embed_size, device):
		super().__init__()
		self.embed = CharLevelEncoder(cvoc, embed_size, device)
		self.rel_embed = RelationEncoder(embed_size)
		self.ff = nn.Sequential(
			nn.Linear(embed_size*3, embed_size),
			nn.LeakyReLU(),
			nn.Linear(embed_size, 2)
		)

	def embed_paths(self, paths: torch.Tensor):
		"""
		:param paths: [l, n, depth, batch]
		:return: [n, depth, batch, embed_size]
		"""
		l, n, depth, batch = paths.shape
		return self.embed(paths.reshape(l, n*depth*batch)).reshape(n, depth, batch, -1)

	def forward(self, h, r, t, paths):
		"""
		:param h: [l, batch]
		:param r: [l, batch]
		:param t: [l, batch]
		:param paths: [l, n, depth, batch]
		:return: score: [batch] probability that r connects h and t
		"""
		# [n, depth, batch, embed_size]
		paths = self.embed_paths(paths)
		# [batch, embed size]
		r = self.embed(r)
		# [batch, embed size]
		r = self.rel_embed(r, paths)
		# [batch, embed size]
		h = self.embed(h)
		# [batch, embed size]
		t = self.embed(t)
		# [batch, embed size*3]
		out = torch.cat([h, r, t], dim=1)
		# [batch, 2]
		out = self.ff(out)
		return out


class APRA(KG):
	"""
	Same as PRA but using Attention
	"""
	module: APRAModule
	optimizer: optim
	graph: Graph3D

	def __init__(self, depth=3, embed_size=64, lr=0.001):
		self.path = "Models/APRA/save.pt"
		if torch.cuda.is_available():
			print("Using the GPU")
			self.device = torch.device("cuda")
		else:
			print("Using the CPU")
			self.device = torch.device("cpu")
		self.depth = max(2, depth)
		self.embed_size = embed_size
		self.lr = lr
		self.batch_size = 2
		self.neg_per_pos = 5
		self.view_every = 0
		self.cmap = IMap("abcdefghijklmnopqrstuvwxyz-/'. ")
		self.loss = nn.CrossEntropyLoss(torch.tensor([1, self.neg_per_pos], dtype=torch.float, device=self.device))

	def epoch(self, it, train=True):
		roll_loss = deque(maxlen=50 if train else None)
		i = 0
		if train and self.view_every:
			plot.plt.ion()
			figw = plot.plt.figure(figsize=(15, 8))
			figg = plot.plt.figure(figsize=(15, 8))
		for h, r, t, paths, labels in it:
			h, r, t, paths, labels = (
				h.to(self.device), r.to(self.device), t.to(self.device),
				paths.to(self.device), labels.to(self.device)
			)
			self.optimizer.zero_grad(set_to_none=True)
			# feed the head and the relation
			preds = self.module(h, r, t, paths)
			# cross entropy
			loss = self.loss(preds, labels)
			# learn
			if train:
				loss.backward()
				if self.view_every and i % self.view_every == 0:
					# noinspection PyUnboundLocalVariable
					figw.clear()
					# noinspection PyUnboundLocalVariable
					figg.clear()
					plot.view_model(figw, self.module, i, grad=False)
					plot.view_model(figg, self.module, i, grad=True)
					figw.canvas.draw()
					figg.canvas.draw()
				i += 1
				self.optimizer.step()
			roll_loss.append(loss.item())
			# display loss
			it.set_postfix_str(
				f"{'' if train else 'val '}loss: {sum(roll_loss)/len(roll_loss):.2f}"
			)
		return sum(roll_loss)/len(roll_loss) if roll_loss else None

	def train(self, train, valid, dataset: str):
		# construct a knowledge graph from training triplets
		self.graph = Graph3D()
		self.graph.add(*train)
		bad_exs = targets(train, dataset)

		def collate_fn(batch):
			hb, rb, tb, pathsb, lb = zip(*batch)
			max_n = max(paths.shape[1] for paths in pathsb)
			max_l = max(paths.shape[0] for paths in pathsb)
			paths = torch.stack([
				pad(paths, (0, 0, 0, max_n-paths.shape[1], 0, max_l-paths.shape[0]))
				for paths in pathsb
			])
			return (
				pad_sequence(hb), pad_sequence(rb),
				pad_sequence(tb), paths.transpose(0, 3), torch.tensor(lb)
			)

		train_batch = data.DataLoader(
			TriplePathCtxData(train, self.graph, self.depth, bad_exs, self.cmap, self.neg_per_pos),
			batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True
		)
		valid_batch = data.DataLoader(
			TriplePathCtxData(valid, self.graph, self.depth, bad_exs, self.cmap, self.neg_per_pos),
			batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=1
		)

		# prepare the model
		self.module = APRAModule(self.cmap, self.embed_size, self.device).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		# train it
		epoch = 1
		best_val = float("+inf")
		patience = 3
		p = patience
		print(f"Early stopping with patience {patience}")
		while epoch <= 1:
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
				torch.save(self.module, self.path)
				best_val = v_loss
				p = patience
			else:
				p -= 1
			epoch += 1
			print()

		# torch.save(self.module, self.path)

	def load(self, train, valid, dataset: str):
		if not os.path.isfile(self.path):
			raise FileNotFoundError("")
		self.graph = Graph3D()
		self.graph.add(*train)
		self.module = APRAModule(self.cmap, self.embed_size, self.device).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		self.module = torch.load(self.path)

	def link_completion(self, n, couples) -> List[str]:
		preds = []
		idx2e = list(self.graph.emap.keys())
		self.module.eval()
		with torch.no_grad():
			for h, r in tqdm(couples, desc="Evaluating", ncols=140):
				# if h is not known, return candidates based on r, and fill randomly if not enough
				candidates = self.graph.r_t[r]
				if h not in self.graph:
					pred = candidates[:n]
				else:
					# for each known t candidates, collect random paths and use the model to score the candidates
					paths = self.graph.random_paths(h, candidates, max_depth=self.depth)
					h = TriplePathCtxData.encode(h, self.cmap).unsqueeze(1)
					r = TriplePathCtxData.encode(r, self.cmap).unsqueeze(1)
					scores = []
					for t in candidates:
						t = TriplePathCtxData.encode(t, self.cmap).unsqueeze(1)
						paths_t = TriplePathCtxData.encode_paths(paths[t], self.cmap, self.depth).unsqueeze(3)
						scores.append(self.module(h, r, t, paths_t)[0].item())

					# score the paths with the model
					# rank them
					pred = [
						idx2e[node] for node, _ in sorted(zip(candidates, scores), key=lambda kv: kv[1], reverse=True)
					][:n]
				# complete with random preds at the end if necessary
				preds.append(pred + [choice(self.graph) for _ in range(n - len(pred))])
		return preds
