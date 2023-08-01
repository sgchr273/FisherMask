import numpy as np
import torch
from .strategy import Strategy
from saving import save_queried_idx

class EntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(EntropySampling, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.savefile = args["savefile"]
		self.alg = "entropy"

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		result = idxs_unlabeled[U.sort()[1][:n]]
		save_queried_idx(result, self.savefile, self.alg)
		return result
