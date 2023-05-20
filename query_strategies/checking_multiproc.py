import torch
from torch.multiprocessing import Pool, Array
import numpy as np
import time
import gc


def initpool(arr):
    global sharedArr
    sharedArr = arr

def trace_for_chunk(xt_, rank, chunkSize, num_gpus, currentInv, fisher, gpu_id):
    total_len = xt_.shape[0] #* num_gpus
    upper_bound = int(total_len/(num_gpus-gpu_id))
    lower_bound = int(upper_bound-(total_len/num_gpus))
    traceEst = np.frombuffer(sharedArr.get_obj(), dtype=np.float32)

    print("Inside trace_for_chunk", total_len, traceEst.shape, lower_bound, upper_bound, chunkSize)
    
    print("Beginning GPU ", gpu_id, " at time: ", time.ctime(), flush=True)
    # for c_idx in range(lower_bound, upper_bound, chunkSize):
    #     xt_chunk = xt_[c_idx : c_idx + chunkSize]
    #     # xt_chunk = torch.tensor(xt_chunk).clone().detach().cuda(gpu_id)
    #     send_time = time.time()
    #     xt_chunk = xt_chunk.clone().detach().cuda(gpu_id)
    #     currentInv = currentInv.cuda(gpu_id)
    #     fisher = fisher.cuda(gpu_id)
    #     print(F"Sent to {gpu_id} in", time.time() - send_time)
    #     innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
        
    #     innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
    #     traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
    #         xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
    #         dim1=-2,
    #         dim2=-1
    #     ).sum(-1).detach().cpu()
    send_time = time.time()
    xt_chunk = xt_[lower_bound : upper_bound]
    xt_chunk = xt_chunk.cuda(gpu_id)
    print(F"Sent to {gpu_id} in", time.time() - send_time)
    fisher = fisher.cuda(gpu_id)
    currentInv = currentInv.cuda(gpu_id)
    
    innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))    
    innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
    traceEst[lower_bound : upper_bound] = torch.diagonal(
        xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
        dim1=-2,
        dim2=-1
    ).sum(-1).detach().cpu()

    # del xt_, fisher, currentInv
    
    print("Finishing GPU ", gpu_id, " at time: ", time.ctime(), flush=True)
    return


if __name__ == '__main__':
    
    rng = np.random.default_rng()
    dim = 10 * 1280 #X.shape[-1]
    rank = 10 #X.shape[-2]
    fisher = torch.rand((dim, dim))
    currentInv = torch.rand((dim, dim))
    xt_ = torch.rand((6000, rank, dim)) #rng.random((2000, rank, dim))
    total_len = xt_.shape[0]
    chunkSize = 500 #len(xt_) // 4 #
    tE = Array('d', total_len, lock=True)
    


    start = time.time()
    xts = [xt_.clone().detach().cuda(0),
           xt_.clone().detach().cuda(1)]
    # xts = [xt_[:total_len//2].clone().detach().cuda(0),
    #        xt_[total_len//2:].clone().detach().cuda(1)]
    mid = time.time()
    # print('Random arrays sent to gpus in', mid - start)

    ## using 2 GPUs
    torch.multiprocessing.set_start_method('spawn', force=True)
    NUM_GPUS = torch.cuda.device_count()
    traceEst = np.frombuffer(tE.get_obj())
    print("Before pool", traceEst.shape)
    with Pool(processes=NUM_GPUS, initializer=initpool, initargs=(tE,)) as pool:
        args = [(xts[x], rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
        # args = [(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
        result = pool.starmap(trace_for_chunk, args)

    with Pool(processes=NUM_GPUS, initializer=initpool, initargs=(tE,)) as pool:
        args = [(xts[x], rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
        # args = [(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
        result = pool.starmap(trace_for_chunk, args)
    
    ## use 1 GPU
    # innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
    # innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
    # traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
    #
    # xt_ = xt_.cuda(0)
    # mid = time.time()
    # NUM_GPUS = 1
    # initpool(tE)
    # trace_for_chunk(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, 0)
    # print(xt_.get_device())
    # trace_for_chunk(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, 0)
    
    traceEst = np.frombuffer(tE.get_obj())
    print(traceEst.shape)
    del fisher, currentInv, xt_, traceEst, tE, xts#, innerInv
    torch.cuda.empty_cache()
    gc.collect()
    print(time.time() - mid, mid - start)