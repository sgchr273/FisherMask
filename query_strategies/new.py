import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Array, Queue, Manager
import numpy as np
import time


num_gpus = 1
rng = np.random.default_rng()
dim = 10 * 700 #X.shape[-1]
rank = 10 #X.shape[-2]
fisher = torch.rand((dim, dim))
currentInv = torch.rand((dim, dim))
xt_ = torch.rand((25000, rank, dim)) #rng.random((2000, rank, dim))
total_len = xt_.shape[0]
chunkSize = 500 #len(xt_) // 4 #
upper_bound = int(total_len/(1-0))
lower_bound = int(upper_bound-(total_len/num_gpus))
xt_chunk = xt_[lower_bound : upper_bound]
xt_chunk = xt_chunk.cuda()
currentInv = currentInv.cuda()

start = time.time()
for i in range(5):
    innerInv = torch.inverse(torch.eye(rank).cuda() + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))

    print(time.time() - start)