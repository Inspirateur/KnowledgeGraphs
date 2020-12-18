from collections import deque, defaultdict
import copy
import functools
from random import choice
from typing import List
import sys
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


class CharLevelEncoder(nn.Module):
	def __init__(self, cvoc, embed_size, device):
		super().__init__()
		self.device = device
		self.embed_size = embed_size
		self.channels = 8
		self.embedding = nn.Embedding(cvoc, self.channels)
		self.conv1 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool1d(3, padding=1)
		self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
		self.pool2 = nn.MaxPool1d(3, padding=1)
		self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
		self.pool3 = nn.MaxPool1d(3, padding=1)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
		self.poolf = nn.AdaptiveAvgPool1d(1)
		self.ff = nn.Linear(128, embed_size)

	def forward(self, seq):
		"""
		Embed the sequence with character level embedding
		:param seq: [l, batch]
		:return: embed: [batch, embed_size]
		"""
		# [batch, channels, l]
		seq = self.embedding(seq.reshape(-1)).reshape(seq.shape[1], -1, seq.shape[0])
		out1 = torch.relu(self.conv1(seq))
		out1p = self.pool1(out1)
		out2 = torch.relu(self.conv2(out1p))
		out2p = self.pool2(out2)
		out3 = torch.relu(self.conv3(out2p))
		out3p = self.pool3(out3)
		out4 = torch.relu(self.conv4(out3p))
		out4p = self.poolf(out4).squeeze()
		outf = torch.relu(self.ff(out4p))
		return outf


class RelationEncoder(nn.Module):
	def __init__(self, embed_size, num_layers=6):
		super().__init__()
		self.path_encoder = nn.GRU(embed_size, embed_size, 2)
		self.layers = nn.TransformerEncoder(
			nn.TransformerEncoderLayer(embed_size, 8), num_layers
		)

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
		# [n+1, batch, embed size]
		src = torch.cat([r.unsqueeze(0), paths], dim=0)
		# pass it through a transformer and retrieve only the representation of r
		# [n+1, batch, embed size]
		src_encoded = self.layers(src)
		# [batch, embed size]
		return src_encoded[0]


# see https://github.com/python/mypy/issues/8795
# noinspection PyAbstractClass
class APRAModule(nn.Module):
	def __init__(self, cvoc, embed_size, device):
		super().__init__()
		self.f1 = nn.Linear(embed_size*3, embed_size)
		self.f2 = nn.Linear(embed_size, 10)
		self.f3 = nn.Linear(10, 2)
		self.embed = CharLevelEncoder(cvoc, embed_size, device)
		self.rel_embed = RelationEncoder(embed_size)

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
		# [batch, embed size]
		out = torch.relu(self.f1(out))
		# [batch, 10]
		out = torch.relu(self.f2(out))
		# [batch, 2]
		out = self.f3(out)
		return out


class APRA(KG):
	"""
	Same as PRA but using Attention
	"""
	module: APRAModule
	optimizer: optim
	graph: Graph3D

	def __init__(self, depth=3, embed_size=128, lr=0.01):
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
		self.cmap = IMap("abcdefghijklmnopqrstuvwxyz-/'. ")
		self.loss = nn.CrossEntropyLoss(torch.tensor([1, self.neg_per_pos], dtype=torch.float, device=self.device))

	def epoch(self, it, train=True):
		roll_loss = deque(maxlen=50 if train else None)
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
			batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=1
		)
		valid_batch = data.DataLoader(
			TriplePathCtxData(valid, self.graph, self.depth, bad_exs, self.cmap, self.neg_per_pos),
			batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=1
		)

		# prepare the model
		self.module = APRAModule(len(self.cmap), self.embed_size, self.device).to(self.device)
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
		torch.save(self.module, self.path)

	def load(self, train, valid, dataset: str):
		self.graph = Graph3D()
		self.graph.add(*train)
		self.module = APRAModule(len(self.cmap), self.embed_size, self.device).to(self.device)
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
				# for each known_candidates
				candidates = {}
				for t in self.graph.r_t[r]:
					# get random paths
					paths = self.graph.random_paths(h, t, self.depth)
					# score the triplet h, r, t
					if not paths:
						candidates[t] = 0
					else:
						candidates[t] = self.module([h], [r], [t], [paths])
				# score the paths with the model
				# rank them
				preds.append([
					idx2e[node] for node, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
				][:n])
		return preds
