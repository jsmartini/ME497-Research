import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import torch
from functools import lru_cache

#jonathan martini
#not optimized for batch evaluation operations
# 1 -> 1

ARGS = namedtuple(
    "ARGS", 
    (
        "LAMBDA",
        "MAXDEPTH",
        "INPUTSZ",
        "OUTPUTSZ",
        "LEVEL"
    )
)

sigmoid = nn.Sigmoid
softmax = nn.Softmax(0)


def recursive_extend(target:list, base_list = []):
    if type(target[0]) != list:
        base_list.extend(target)
        #print(base_list)
        return base_list
    elif type(target[0]) == list:
        return list(map(lambda l: recursive_extend(l, base_list), target))


class Inner(nn.Module):
    
    inner_collection = []

    def __init__(self, args: ARGS):
        super(Inner, self).__init__()
        self.args = args
        self.fc = nn.Linear(self.args.INPUTSZ, 1)
        self.Beta = torch.randn(1)
        #tracks beta on autograd
        self.BetaParam = nn.Parameter(self.Beta, requires_grad=True)
        next_level_args = ARGS(args.PENALTY, args.MAXDEPTH, args.INPUTSZ, args.OUTPUTSZ, args.LEVEL+1)
        if args.LEVEL != args.MAXDEPTH:
            self.L = Inner(next_level_args)
            self.R = Inner(next_level_args)
        elif args.LEVEL == args.MAXDEPTH:
            self.L = Leaf(next_level_args)
            self.R = Leaf(next_level_args)

        self.inner_collection[args.LEVEL].append(self)             #make object reference visible globally in the tree

    def getParameters(self):
        return [super().parameters()] + self.R.getParameters() + self.L.getParameters()

    def _pforward(self, x, P):
        out = sigmoid(self.Beta*self.fc(x))
        return out, out*p

    def _penalty(self):
        self.penalty_log_computed = 0.5*(torch.log(self.penalty) + torch.log(1-self.self.penalty))

    def forward(self, x, P = 1, train=False):
        self.prob = P
        out, p = self._pforward(x, p)
        if train:
            self.penalty = p    #no batch size so it is just P (sums cancel)
            self._penalty()
            if p > 0:
                self.R(x, p = (1-p*P), train = train)
                return self.L(x, p = p*P, train = train)
            elif p < 0:
                self.L(x, p = (1-p*P), train = train)
                return self.R(x, p = p*P, train = train)
        else:
            #only returns the dist with highest path prob
            if p > 0:           
                return self.L(x, p = (1-P*p), train = train)[0]
            elif p < 0:
                return self.R(x, p = P*p, train = train)[0]

    def __call__(self, x, p=1, train = False):
        return self.forward(x, p=p, train = False)


class Leaf(nn.Module):

    leaf_collection = []

    def __init__(self, args):
        super(Leaf, self).__init__()
        self.fc = nn.Linear(args.OUTPUTSZ, 1)
        self.Beta = torch.randn(1)
        self.args = args
        #tracks beta on autograd
        self.leaf_collection.append(self)   #register object with the rest of the leaf pool

    def collect_probs(self):
        return torch.Tensor([node.prob for node in self.leaf_collection])

    def collect_dists(self):
        return torch.cat([node.dist for node in self.leaf_collection])

    def __call__(self, x, p, train=False):
        self.prob = p
        out = softmax(self.fc(x))
        self.dist = out
        return out, p, self                 #returns most probable distribution, path prob and object reference

    def getParameters(self):
        #get parameters
        return super().parameters()

class QSDT(nn.Module):

    def __init__(self, args: ARGS):
        self.args = args
        self.root = Inner(args)
        self.root.inner_collection = [[] for _ in range(args.MAXDEPTH-1)]

    def __call__(self, x, train = False):
        return self.root(x, p=1, train=train)

    def Loss(self, x, y):
        pred, pmax, node = self(x, train = True)
        all_probs = node.collect_probs()
        all_dists = node.collect_dists()
        y = torch.cat([y for _ in range(all_dists.shape[0])])
        y = torch.autograd.Variable(y)
        return torch.sum(
            torch.dot(
                all_probs,
                torch.dot(y, torch.log(all_dists), dim=1),
                dim = 0
            )
        )

    @lru_cache(maxsize=25)                                      #only computes certain things once so computation is faster i.e. 2**-lmda
    def _penalize(self):
        tree = self.root.inner_collection           #reference to organized tree heirarchy
        with torch.no_grad():
            s = torch.sum(recursive_extend([[node.penalty_log_computed*(2**-node.args.LEVEL) for node in tree[level]] for level in range(len(self.args.MAXDEPTH-2))]))
                




    
        

        