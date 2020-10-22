from Models.Random.random import Random
from Models.PRA.pra import PRA
from Models.Proximity.proximity import Proximity
from Models.TransE.transe import TransE
from Models.GAT.gat import GAT
from datasets import WIKI, WN18RR, FB15k
from Models.KG import eval_link_completion, hyper_train, browse_mistakes


n = 100
eval_link_completion(n, FB15k, PRA())
# browse_mistakes(n, FB15k, TransE())
