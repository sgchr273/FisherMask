import torch
import time
import numpy as np
from torch.multiprocessing import Pool, current_process, Queue
NUM_GPUS = 4
  
chunkSize = 100


def trace_for_chunk(c_idx, xt_, chunkSize, queue, currentInv, fisher):
    gpu_id = queue.get()
    print("Entering", gpu_id)
    xt_chunk = xt_[c_idx : c_idx + chunkSize]
    xt_chunk = xt_chunk.clone().detach().cuda(gpu_id)
    currentInv = currentInv.cuda(gpu_id)
    fisher = fisher.cuda(gpu_id)
    innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
    trace = torch.diagonal(
        xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2),
        dim1=-2, 
        dim2=-1
    ).sum(-1).detach().cpu()
    queue.put(gpu_id)
    print("Exiting", gpu_id)
    return trace

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force = True)    
    with Pool(processes=NUM_GPUS) as pool:

        m = torch.multiprocessing.Manager()
        queue = m.Queue()
        #queue = Queue()
        
        for gpu_ids in range(NUM_GPUS):
            queue.put(gpu_ids)

        xt_ = torch.rand(50000, 10, 15000)
        rank = xt_.shape[-2]
        random_mat = torch.rand(xt_.shape[-1], xt_.shape[-1])
        random_mat2 = torch.rand(xt_.shape[-1], xt_.shape[-1])
        currentInv = torch.inverse(random_mat @ random_mat.T)
        fisher = (random_mat2 @ random_mat2.T)

        args = [(c_idx, xt_, chunkSize, queue, currentInv, fisher) for c_idx in range(0, xt_.shape[0], chunkSize)]

        result = pool.starmap(trace_for_chunk, args)


