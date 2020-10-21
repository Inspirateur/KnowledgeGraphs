import itertools
from typing import List, Tuple
import json
import readline
from datasets import load_dataset
from Structures.graph3d import Graph3D


class KG:
	def train(self, train, valid, dataset: str):
		raise NotImplementedError("")

	def load(self, train, valid, dataset: str):
		raise NotImplementedError("")

	def link_completion(self, n, couples) -> List[str]:
		raise NotImplementedError("")

	def eval_link_completion(self, n, test):
		test_x = []
		test_y = []
		for h, r, t in test:
			test_x.append((h, r))
			test_y.append(t)
		preds = self.link_completion(n, test_x)
		assert len(test_y) == len(preds)
		acc = None
		if len(test_y):
			acc = sum(is_top_n(n, t, p) for t, p in zip(test_y, preds))/len(test_y)
		return acc


def is_top_n(n, target, preds):
	for i in range(min(n, len(preds))):
		if preds[i] == target:
			return True
	return False


def hyper_train(n, dataset: str, kg_class: KG.__class__, hypers: dict):
	keys = list(hypers.keys())
	res = {"keys": keys, "values": []}
	best_acc = 0
	best_params = None
	for values in itertools.product(*hypers.values()):
		param_dict = {k: v for k, v in zip(keys, values)}
		print(", ".join(f"{k}={v}" for k, v in zip(keys, values)))
		acc = eval_link_completion(n, dataset, kg_class(**param_dict))
		res["values"].append(list(values) + [acc])
		if acc > best_acc:
			best_acc = acc
			best_params = values
		print()
	assert best_params is not None
	txt_params = ", ".join(f"{k}={v}" for k, v in zip(keys, best_params))
	print(f"Best acc = {best_acc:.2%} with params {txt_params}")
	with open(f"{KG.__name__}_results.json", "w") as fres:
		json.dump(res, fres)


def eval_link_completion(n, dataset: str, kg: KG):
	print(f"Evaluating {kg.__class__.__name__} KG on link completion:")
	train, valid, test = load_dataset(dataset)
	try:
		kg.load(train, valid, dataset)
	except FileNotFoundError:
		kg.train(train, valid, dataset)
	acc = kg.eval_link_completion(n, test)
	print(f"Hit@{n} Accuracy = {acc:.2%}")
	return acc


def browse_mistakes(n, dataset: str, kg: KG):
	print(f"Browsing Hit@{n} mistakes of {kg.__class__.__name__} KG on link completion :")
	train, valid, test = load_dataset(dataset)
	graph = Graph3D()
	graph.add(*train)
	# kg.train(train, valid)
	kg.load(train, valid)

	# get all mistakes made by the model
	print("Gathering every mistakes...", end=" ")
	test_x = []
	test_y = []
	for h, r, t in test:
		test_x.append((h, r))
		test_y.append(t)
	preds = kg.link_completion(n, test_x)
	mistakes = [(h, r, t, p) for (h, r), t, p in zip(test_x, test_y, preds) if not is_top_n(n, t, p)]
	print(f"Done ({1-len(mistakes)/len(test_y):.2%} accuracy)\n")
	batch = 5
	idx = 0
	entities = []

	def completer(text, state):
		options = [e for e in entities if e.startswith(text)]
		return options[state] if state < len(options) else None

	readline.parse_and_bind("tab: complete")
	readline.set_completer(completer)

	while True:
		print(f"Batch of {batch} mistakes:")
		entities.clear()
		for i in range(min(batch, len(mistakes)-idx)):
			h, r, t, p = mistakes[idx+i]
			entities.append(h)
			entities.append(t)
			print(f"{h} {r} -> {t}")
			print("\t" + ", ".join(p))
			print()
		idx += batch % len(mistakes)
		while True:
			print("\ntype n for next batch or <entity> to explore the entity")
			inpt = input("")
			print()
			if inpt.lower() in {"n", "next"}:
				break
			if inpt in graph:
				print(inpt, ":")
				for r, t in graph.inspect(inpt):
					print(r, t)
			else:
				print(f"{inpt} is not known in the knowledge graph")
