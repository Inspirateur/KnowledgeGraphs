from collections import deque
from math import sqrt
from typing import List, Tuple
import sys
import torch
from torch.utils import data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from tqdm import tqdm
from Models.KG import KG
from Structures.graph3d import Graph3D
from Structures.triplet_dataset import TripleNData


class GraphAttentionHead(nn.Module):
	def __init__(self, in_dim, out_dim):
		nn.Module.__init__(self)
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.sqrtdk = sqrt(out_dim)
		self.Q = nn.Linear(in_dim, out_dim)
		self.K = nn.Linear(in_dim, out_dim)
		self.V = nn.Linear(in_dim, out_dim)

	def forward(self, x: torch.Tensor, xq: torch.Tensor):
		"""
		Compute attention between a node and its neighbors, for knowledge graphs
		:param x: [B x dim] the nodes
		:param xq: [n x B x dim] the neighbors
		:return: [B x dk] a smaller representation of x
		"""
		q = self.Q(x).unsqueeze(2)  # [B x dk x 1]
		k = self.K(xq).transpose(0, 1)  # [B x n x dk]
		v = self.V(xq)  # [n x B x dk]
		att = torch.bmm(k, q).view(-1, x.size()[0])/self.sqrtdk  # [n x B]
		res = f.softmax(att, dim=0)[:, :, None]*v  # [n x B x dk]
		return res.sum(dim=0)  # [B x dk]


class MultiheadGraphAttention(nn.Module):
	def __init__(self, dim, nheads):
		nn.Module.__init__(self)
		assert dim % nheads == 0
		self.dim = dim
		self.nheads = nheads
		self.attentions = [
			GraphAttentionHead(dim, dim//nheads)
			for _ in range(nheads)
		]
		for i, attention in enumerate(self.attentions):
			self.add_module(f"attention_{i}", attention)
		self.out = nn.Linear(dim, dim)

	def forward(self, x, xq):
		atts = torch.cat(tuple(att(x, xq) for att in self.attentions), dim=1)
		return self.out(atts)


class GATModule(nn.Module):
	def __init__(self, v_e, v_r, dim, nheads=5):
		nn.Module.__init__(self)
		urange = 6 / dim ** .5
		self.dim = dim
		self.e_embed = nn.Embedding(v_e, dim)
		self.final_e_embed = nn.Embedding(v_e, dim)
		nn.init.uniform_(self.e_embed.weight.data, -urange, urange)
		self.r_embed = nn.Embedding(v_r, dim)
		self.final_r_embed = nn.Embedding(v_r, dim)
		nn.init.uniform_(self.r_embed.weight.data, -urange, urange)
		self.attention = MultiheadGraphAttention(3*dim, nheads)

	def embed_e(self, entities):
		return self.e_embed(entities.reshape(-1)).view(*entities.shape, -1)

	def embed_r(self, relations):
		return (torch.sign(relations).reshape(-1)[:, None]*self.e_embed(torch.abs(relations.reshape(-1)))).view(*relations.shape, -1)

	def embed_triples(self, triples):
		return torch.cat(
			[self.embed_e(triples[0, :]), self.embed_r(triples[1, :]), self.embed_e(triples[2, :])],
			dim=-1
		)

	def forward(self, pos_triples, neg_triples, pos_neigh, neg_neigh):
		self.e_embed.data = f.normalize(self.e_embed.weight.data).detach()
		# compute positive example embedding
		pos_triples_att = self.attention(self.embed_triples(pos_triples), self.embed_triples(pos_neigh))
		ph, pr, pt = pos_triples_att[:, :self.dim], pos_triples_att[:, self.dim:self.dim*2], pos_triples_att[:, self.dim*2:]
		# compute negative example embedding
		neg_triples_att = self.attention(self.embed_triples(neg_triples), self.embed_triples(neg_neigh))
		nh, nr, nt = neg_triples_att[:, :self.dim], neg_triples_att[:, self.dim:self.dim*2], neg_triples_att[:, self.dim*2:]
		# store the final embeddings
		with torch.no_grad():
			self.final_e_embed.weight[pos_triples[0, :]] = ph
			self.final_r_embed.weight[pos_triples[1, :]] = pr
			self.final_e_embed.weight[pos_triples[2, :]] = pt
			self.final_e_embed.weight[neg_triples[0, :]] = nh
			self.final_r_embed.weight[neg_triples[1, :]] = nr
			self.final_e_embed.weight[neg_triples[2, :]] = nt
		# return positive and negative distances (for the loss)
		return (ph+pr-pt).norm(dim=1), (nh+nr-nt).norm(dim=1)


class GAT(KG):
	"""
	Graph network with attention, https://arxiv.org/pdf/1906.01195.pdf (not the actual GAT but it's based on it)
	See also https://github.com/deepakn97/relationPrediction author's github
	"""
	module: GATModule
	optimizer: optim
	graph: Graph3D
	device: torch.device

	def __init__(self, lr=0.005,  margin=2, dim=50):
		self.batch_size = 128
		self.path = "Models/GAT/save.pt"
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

	def epoch(self, it, learn=True):
		roll_loss = deque(maxlen=50 if learn else None)
		roll_pd = deque(maxlen=50 if learn else None)
		roll_nd = deque(maxlen=50 if learn else None)
		for pos_triples, neg_triples, pos_neigh, neg_neigh in it:
			pos_triples = torch.stack(pos_triples).to(torch.long).to(self.device)
			neg_triples = torch.stack(neg_triples).to(torch.long).to(self.device)
			pos_neigh = torch.stack(
				[torch.stack(triples) for triples in pos_neigh]
			).transpose(0, 1).to(torch.long).to(self.device)
			neg_neigh = torch.stack(
				[torch.stack(triples) for triples in neg_neigh]
			).transpose(0, 1).to(torch.long).to(self.device)
			self.optimizer.zero_grad()
			# feed the head and the relation
			pos_dist, neg_dist = self.module(
				pos_triples, neg_triples, pos_neigh, neg_neigh
			)
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

	def load(self, train, valid, dataset: str):
		self.graph = Graph3D()
		self.graph.add(*train)
		self.module = torch.load(self.path)
		self.optimizer = optim.Adam(self.module.parameters(), lr=self.lr)
		valid_batch = data.DataLoader(TripleNData(valid, self.graph), batch_size=self.batch_size)
		valid_it = tqdm(valid_batch, desc="\tValidating", ncols=140, file=sys.stdout)
		with torch.no_grad():
			self.epoch(valid_it, learn=False)

	def train(self, train, valid, dataset: str):
		# prepare the data
		self.graph = Graph3D()
		self.graph.add(*train)
		train_batch = data.DataLoader(TripleNData(train, self.graph), batch_size=self.batch_size)
		valid_batch = data.DataLoader(TripleNData(valid, self.graph), batch_size=self.batch_size)

		# prepare the model
		self.module = GATModule(len(self.graph.emap), len(self.graph.rmap), dim=self.dim).to(self.device)
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
			train_it = tqdm(train_batch, desc="\tTraining", ncols=140, file=sys.stdout)
			self.epoch(train_it)

			# validation
			self.module.eval()
			valid_it = tqdm(valid_batch, desc="\tValidating", ncols=140, file=sys.stdout)
			with torch.no_grad():
				v_loss, v_pd, v_nd = self.epoch(valid_it, learn=False)
			if v_loss < best_val:
				torch.save(self.module, self.path)
				best_val = v_loss
				p = patience
			else:
				p -= 1
			epoch += 1
			print()
		print(f"Loading best val loss = {best_val:.2f} at epoch {epoch-patience-1}")
		# self.module = torch.load(self.path)
		self.inspect_embeddings()

	def link_completion(self, n, couples) -> List[List[Tuple[str, int]]]:
		preds = []
		idx2e = list(self.graph.emap.keys())
		self.module.eval()
		with torch.no_grad():
			for h, r in couples:
				# get predictions
				h = torch.tensor([self.graph.emap[h]], device=self.device)
				r = torch.tensor([self.graph.rmap[r]], device=self.device)
				d = self.module.final_e_embed(h) + self.module.r_embed(r)
				# find the closest embeddings
				distances = torch.norm(self.module.final_e_embed.weight-d.view(-1)[None, :], dim=1)
				vals, indices = distances.topk(k=n, largest=False)
				preds.append([idx2e[i] for i in indices.flatten().tolist()])
		return preds
