from collections import deque, defaultdict
from typing import List, Tuple
import sys
import torch
from torch.utils import data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from tqdm import tqdm
from Models.KG import KG
from Structures.imap import IMap
from Structures.triplet_dataset import TripleData
from visualisation import plot_entity


# see https://github.com/python/mypy/issues/8795
# noinspection PyAbstractClass
class TransEModule(nn.Module):
	def __init__(self, v_e, v_r, dim):
		nn.Module.__init__(self)
		urange = 6/dim**.5
		self.e_embed = nn.Embedding(v_e, dim)
		nn.init.uniform_(self.e_embed.weight.data, -urange, urange)
		self.r_embed = nn.Embedding(v_r, dim)
		nn.init.uniform_(self.r_embed.weight.data, -urange, urange)
		self.r_embed.data = f.normalize(self.r_embed.weight.data, p=1).detach()

	def forward(self, pos_triples, neg_triples):
		self.e_embed.data = f.normalize(self.e_embed.weight.data)
		pos_dist = self.distance(pos_triples)
		neg_dist = self.distance(neg_triples)
		return pos_dist, neg_dist

	def distance(self, triples):
		return (self.e_embed(triples[0, :])+self.r_embed(triples[1, :])-self.e_embed(triples[2, :])).norm(p=1, dim=1)


class TransE(KG):
	"""
	A reimplementation of TransE which learns embedding for h and r such that h + r ≈ t
	"""
	module: TransEModule
	optimizer: optim
	emap: IMap
	rmap: IMap
	h2t: dict
	device: torch.device

	def __init__(self, lr=0.005,  margin=45, dim=50):
		self.path = "Models/TransE/save.pt"
		self.batch_size = 128
		if torch.cuda.is_available():
			print("Using the GPU")
			self.device = torch.device("cuda")
		else:
			print("Using the CPU")
			self.device = torch.device("cpu")
		# hyperparameters
		self.lr = lr
		self.margin = margin
		self.dim = dim
		self.limit = 7

	def inspect_embeddings(self):
		e_avg = self.module.e_embed.weight.mean(dim=0)
		e_var = (e_avg-self.module.e_embed.weight).norm(dim=1).mean()
		print(
			f"E avg norm {e_avg.norm():.2f}, E var {e_var:.2f}, "
			f"R norm avg {self.module.r_embed.weight.norm(dim=1).mean():.2f}"
		)
		plot_entity(self.module.e_embed.weight.cpu().detach().numpy(), self.emap)

	def epoch(self, it, learn=True):
		roll_loss = deque(maxlen=50 if learn else None)
		roll_pd = deque(maxlen=50 if learn else None)
		roll_nd = deque(maxlen=50 if learn else None)
		for pos_triples, neg_triples in it:
			pos_triples = torch.stack(pos_triples).to(torch.long).to(self.device)
			neg_triples = torch.stack(neg_triples).to(torch.long).to(self.device)
			self.optimizer.zero_grad()
			# feed the head and the relation
			pos_dist, neg_dist = self.module(pos_triples, neg_triples)
			loss = self.criterion(pos_dist, neg_dist)
			roll_pd.append(pos_dist.mean())
			roll_nd.append(neg_dist.mean())
			# learn
			if learn:
				loss.backward()
				self.optimizer.step()
			roll_loss.append(loss.item())
			# display loss
			it.set_postfix_str(
				f"{'' if learn else 'val '}loss: {sum(roll_loss)/len(roll_loss):.2f}, "
				f"pos dist: {sum(roll_pd)/len(roll_pd):.2f}, "
				f"neg dist: {sum(roll_nd)/len(roll_nd):.2f}"
			)
		return sum(roll_loss)/len(roll_loss), sum(roll_pd)/len(roll_pd), sum(roll_nd)/len(roll_nd)

	def criterion(self, pd, nd):
		return torch.clamp_min(pd-nd+self.margin, 0).mean()

	def train(self, train, valid, dataset: str):
		path = "Models/TransE/save.pt"

		# prepare the data
		self.emap = IMap()
		self.rmap = IMap()
		self.h2t = defaultdict(list)
		for h, r, t in train:
			self.emap.put(h)
			self.emap.put(t)
			self.rmap.put(r)
			self.h2t[h].append(self.emap[t])
		for h, tails in self.h2t.items():
			self.h2t[h] = torch.tensor(tails)
		train_batch = data.DataLoader(TripleData(train, self.emap, self.rmap), batch_size=self.batch_size)
		valid_batch = data.DataLoader(TripleData(valid, self.emap, self.rmap), batch_size=self.batch_size)

		# prepare the model
		self.module = TransEModule(len(self.emap), len(self.rmap), dim=self.dim).to(self.device)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)

		# train it
		epoch = 1
		best_val = float("+inf")
		patience = 5
		p = patience
		print(f"Early stopping with patience {patience}")
		while p > 0:
			print(f"Epoch {epoch}")

			# training
			self.module.train()
			train_it = tqdm(train_batch, desc="\tTraining", file=sys.stdout)
			self.epoch(train_it)

			# validation
			self.module.eval()
			valid_it = tqdm(valid_batch, desc="\tValidating", file=sys.stdout)
			with torch.no_grad():
				v_loss, v_pd, v_nd = self.epoch(valid_it, learn=False)
			if v_loss < best_val:
				torch.save(self.module, path)
				best_val = v_loss
				p = patience
			else:
				p -= 1
			epoch += 1
			print()
		print(f"Loading best val loss = {best_val:.2f} at epoch {epoch-patience-1}")
		# self.module = torch.load(path)
		self.inspect_embeddings()

	def load(self, train, valid, dataset: str):
		# prepare the data
		self.emap = IMap()
		self.rmap = IMap()
		self.h2t = defaultdict(list)
		for h, r, t in train:
			self.emap.put(h)
			self.emap.put(t)
			self.rmap.put(r)
			self.h2t[h].append(self.emap[t])
		for h, tails in self.h2t.items():
			self.h2t[h] = torch.tensor(tails)
		self.module = torch.load(self.path)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		valid_batch = data.DataLoader(TripleData(valid, self.emap, self.rmap), batch_size=self.batch_size)
		valid_it = tqdm(valid_batch, ncols=140, desc="\tValidating", file=sys.stdout)
		with torch.no_grad():
			self.epoch(valid_it, learn=False)
		self.inspect_embeddings()

	def link_completion(self, n, couples) -> List[List[Tuple[str, int]]]:
		preds = []
		idx2e = list(self.emap.keys())
		self.module.eval()
		with torch.no_grad():
			for h, r in couples:
				# get predictions
				hid = torch.tensor([self.emap[h]], device=self.device)
				rid = torch.tensor([self.rmap[r]], device=self.device)
				d = self.module.e_embed(hid)+self.module.r_embed(rid)
				# find the closest embeddings
				distances = torch.norm(self.module.e_embed.weight-d.view(-1)[None, :], dim=1)
				# filter out the direct connexions to boost accuracy
				distances[self.h2t[h]] = float("+inf")
				vals, indices = distances.topk(k=n, largest=False)
				preds.append([idx2e[i] for i in indices.flatten().tolist()])
		return preds
