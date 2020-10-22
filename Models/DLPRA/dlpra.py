from typing import Dict, List, Tuple
import torch
from torch.utils import data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from Models.KG import KG


class AttentionHead(nn.Module):
	def __init__(self, in_dim, out_dim):
		nn.Module.__init__(self)
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.Q = nn.Linear(in_dim, out_dim)
		self.K = nn.Linear(in_dim, out_dim)

	def forward(self, x: torch.Tensor, xq: torch.Tensor):
		"""
		:param x: [B x dim] The relation
		:param xq: [n x B x dim] the paths
		:return: [n x B] attention's scores
		"""
		q = self.Q(x).unsqueeze(2)  # [B x dk x 1]
		k = self.K(xq).transpose(0, 1)  # [B x n x dk]
		att = torch.bmm(k, q).view(-1, x.size()[0])/self.sqrtdk  # [n x B]
		return f.softmax(att, dim=0)  # [n x B]


class DLPRA(KG):
	"""
	Same as PRA but using deep learning, more formally:
	Given r and a distribution of targets for each path predict t,
	using r as a query, and all paths as context we can use attention to give weight to paths on the fly
	"""
	# <relation, <path, score>>
	paths: Dict[int, Dict[Tuple, float]]

	def train(self, train, valid, dataset: str):
		pass

	def load(self, train, valid, dataset: str):
		pass

	def link_completion(self, n, couples) -> List[str]:
		pass
