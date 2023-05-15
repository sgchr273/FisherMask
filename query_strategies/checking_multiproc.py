import torch
from torch.multiprocessing import Pool, Array
import numpy as np
import time


def initpool(arr):
    global sharedArr
    sharedArr = arr

def trace_for_chunk(xt_, rank, chunkSize, num_gpus, currentInv, fisher, gpu_id):
    upper_bound = int(xt_.shape[0]/(num_gpus-gpu_id))
    lower_bound = int(upper_bound-(xt_.shape[0]/num_gpus))
    t = time.time()
    traceEst = np.frombuffer(sharedArr.get_obj(), dtype=np.float64)
    # print("Beginning GPU ", gpu_id, " at time: ", time.time(), flush=True)
    for c_idx in range(lower_bound, upper_bound, chunkSize):
        xt_chunk = xt_[c_idx : c_idx + chunkSize]
        # xt_chunk = torch.tensor(xt_chunk).clone().detach().cuda(gpu_id)
        xt_chunk = xt_chunk.clone().detach().cuda(gpu_id)
        currentInv = currentInv.cuda(gpu_id)
        fisher = fisher.cuda(gpu_id)
        innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
        # print('fisher: ', fisher, '\n', 'curentinv: ', currentInv, '\n', 'xt_chunk: ', xt_chunk)
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
            xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1
        ).sum(-1).detach().cpu()
        # print(time.time()-t)
    del xt_chunk, fisher, currentInv
    # print("Finishing GPU ", gpu_id, " at time: ", time.time(), flush=True)
    return


