from random import choices
from Models.KG import KG


class Random(KG):
	"""
	A baseline Knowledge Graph that does link completion at random
	"""
	emap: dict

	def __init__(self):
		self.limit = 7

	def load(self, train, valid, dataset: str):
		self.train(train, valid, dataset)

	def train(self, train, valid, dataset: str):
		# map entities to an id
		self.emap = {}
		for h, _, t in train:
			if h not in self.emap:
				self.emap[h] = len(self.emap)
			if t not in self.emap:
				self.emap[t] = len(self.emap)

	def link_completion(self, n, doubles):
		entities = list(self.emap.keys())[1:]
		preds = []
		for _ in doubles:
			preds.append(list(choices(entities, k=n)))
		return preds
