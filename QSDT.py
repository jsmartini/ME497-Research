import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import torch
from functools import lru_cache
from itertools import chain
from torch.optim import Optimizer
from dataclasses import dataclass
from numpy import log
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

class Inner(nn.Module):
    
    inner_collection = []

    def __init__(self, args: ARGS):
        super(Inner, self).__init__()
        self.args = args
        self.fc = nn.Linear(self.args.INPUTSZ, 1)
        #self.Beta = torch.randn(1)
        self.penalty = 0
        self.prob = 0
        self.penalty_log_computed = 0
        #tracks beta on autograd
        #self.BetaParam = nn.Parameter(self.Beta, requires_grad=True)
        next_level_args = ARGS(args.LAMBDA, args.MAXDEPTH, args.INPUTSZ, args.OUTPUTSZ, args.LEVEL+1)
        if args.LEVEL != args.MAXDEPTH:
            self.L = Inner(next_level_args)
            self.R = Inner(next_level_args)
        elif args.LEVEL == args.MAXDEPTH:
            self.L = Leaf(next_level_args)
            self.R = Leaf(next_level_args)

    def link(self):
       #turns tree ref into heap
       self.inner_collection.append(self)
       self.R.link()
       self.L.link()
       
    def getParameters(self):
        return [super().parameters()] + self.R.getParameters() + self.L.getParameters()

    def _pforward(self, x, P):
        out = sigmoid(self.fc(x))
        return out, out*P

    def _penalty(self):
        self.penalty_log_computed = 0.5*(log(self.penalty) + log(1-self.penalty))

    def forward(self, x, P = 1, train=False):
        self.prob = P
        out, p = self._pforward(x, p)
        if train:
            self.penalty = p    #no batch size so it is just P (sums cancel)
            self._penalty()
            if p >= 0.5:
                self.R(x, p = (1-p*P), train = True)
                return self.L(x, p = p*P, train = True)
            elif p < 0.5:
                self.L(x, p = (1-p*P))
                return self.R(x, p = p*P)
        else:
            #only returns the dist with highest path prob
            if p >= 0.5:           
                return self.L(x, p = (1-P*p), train = train)[0]
            elif p < 0.5:
                return self.R(x, p = P*p, train = train)[0]

    def __call__(self, x, p=1, train = False):
        return self.forward(x, p=p, train = False)



class Leaf(nn.Module):

    leaf_collection = []

    def __init__(self, args):
        super().__init__()
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

    def link(self, l = 0):
        pass

class QSDT(nn.Module):

    def __init__(self, args: ARGS):
        super(QSDT, self).__init__()
        self.args = args
        self.root = Inner(args)
        self.root.link()
        self.linear_params = [p.view(-1) for p in self.root.getParameters()]

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

    def fit(self, dset: DataLoader,opt: Optimizer, epochs: int):
        for epoch in range(epochs):
            epoch_loss = 0
            for idx, (x, y) in enumerate(dset):
                opt.zero_grad()
                loss = self.Loss(x, y) + self.penalize()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            print(f"{epoch+1}/{epochs}:\tAVG Loss {epoch_loss/len(dset)}")


    def penalize(self):
        with torch.no_grad():
            #heap traversal to calculate tree layer penalties
            C = torch.sum(torch.Tensor(list(chain(*[
                [2**(-n) * node.penalty_log_computed for node in self.root.inner_collection[2**n-1:2**(n+1)-1]] for n in range(self.args.MAXDEPTH-1)
            ]))))
            return C #+ torch.norm(self.linear_params, 1)




    
        


        