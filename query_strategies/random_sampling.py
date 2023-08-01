import numpy as np
from .strategy import Strategy
import pdb
from saving import  save_queried_idx

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.alg = "rand"
        self.savefile = args["savefile"]

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        chosen = inds[np.random.permutation(len(inds))][:n]
        save_queried_idx(chosen, self.savefile, self.alg)
        return chosen
