import json
from random import random
from tqdm import tqdm

WIKI = "Wiki"
WN18RR = "WN18RR"
FB15k = "FB15K-237"


def load_dataset(d: int):
	if d == WIKI:
		train, valid, test = [], [], []
		split = .001
		with open("Datasets/Wiki/infobox_en_clean.ttl", "r") as ftrain:
			for line in tqdm(ftrain, desc="Wikiboxes", total=14_834_878):
				if random() > split:
					train.append(line.strip().split())
				else:
					if random() > .5:
						test.append(line.strip().split())
					else:
						valid.append(line.strip().split())
			return train, valid, test
	if d == WN18RR:
		with open("Datasets/WN18RR/train.txt", "r") as ftrain:
			train = [line.strip().split("\t") for line in ftrain]
		with open("Datasets/WN18RR/valid.txt", "r") as fvalid:
			valid = [line.strip().split("\t") for line in fvalid]
		with open("Datasets/WN18RR/test.txt", "r") as ftest:
			test = [line.strip().split("\t") for line in ftest]
		return train, valid, test
	if d == FB15k:
		with open("Datasets/FB15k-237/entity2wikidata.json", "r") as fen2wiki:
			enw2wiki: dict = json.load(fen2wiki)
		labels = {}
		for eid, info in enw2wiki.items():
			labels[eid] = info["wikipedia"].rsplit("wiki/", 1)[-1].replace("_", " ") if info["wikipedia"] else info["label"]

		def labelize(h, r, t):
			if h in labels:
				h = labels[h]
			if t in labels:
				t = labels[t]
			return h, r, t

		with open("Datasets/FB15k-237/train.txt", "r") as ftrain:
			train = [labelize(*line.strip().split("\t")) for line in ftrain]
		with open("Datasets/FB15k-237/valid.txt", "r") as fvalid:
			valid = [labelize(*line.strip().split("\t")) for line in fvalid]
		with open("Datasets/FB15k-237/test.txt", "r") as ftest:
			test = [labelize(*line.strip().split("\t")) for line in ftest]
		return train, valid, test
	return [], [], []
