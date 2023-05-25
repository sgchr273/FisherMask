import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Array, Queue, Manager
import numpy as np
import time
import gc


def initpool(arr):
    global sharedArr
    sharedArr = arr

def trace_for_chunk(xt_, rank, chunkSize, num_gpus,currentInv,fisher, gpu_id ):
    total_len = xt_.shape[0] #* num_gpus
    upper_bound = int(total_len/(num_gpus-gpu_id))
    lower_bound = int(upper_bound-(total_len/num_gpus))
    traceEst = np.frombuffer(sharedArr.get_obj(), dtype=np.float32)

    print("Inside trace_for_chunk", total_len, traceEst.shape, lower_bound, upper_bound, chunkSize)
    
    print("Beginning GPU ", gpu_id, " at time: ", time.ctime(), flush=True)
    xt_chunk = xt_[lower_bound : upper_bound]
    # queue.put(xt_chunk)
    # queue.put(fisher)
    # queue.put(currentInv)
    xt_chunk = xt_chunk.cuda(gpu_id)
    fisher = fisher.cuda(gpu_id)
    currentInv = currentInv.cuda(gpu_id)
    # print(F"Sent to {gpu_id} in", time.time() - send_time)
    # with torch.no_grad():
    innerInv_time = time.time()
    innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2)) 
    print("InnerInv at GPU", innerInv.get_device(), " took ", time.time()-innerInv_time, "has shape", innerInv.shape)
    tracest_time = time.time()
    traceEst[lower_bound : upper_bound] = torch.diagonal(
        xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
        dim1=-2,
        dim2=-1
    ).sum(-1).detach().cpu()

    del xt_, fisher, currentInv, innerInv
    print("traceEst calculated in", time.time() - tracest_time)
    # print("Finishing GPU ", gpu_id, " at time: ", time.ctime(), flush=True)
    return traceEst

    
if __name__ == '__main__':
    
    rng = np.random.default_rng()
    dim = 10 * 780 #X.shape[-1]
    rank = 10 #X.shape[-2]
    fisher = torch.rand((dim, dim))
    currentInv = torch.rand((dim, dim))
    xt_ = torch.rand((1000, rank, dim)) #rng.random((2000, rank, dim))
    total_len = xt_.shape[0]
    chunkSize = 500 #len(xt_) // 4 #
    tE = Array('d', total_len, lock=True)
    


    start = time.time()
    fishers = [fisher.clone().detach().cuda(0), fisher.clone().detach().cuda(1)]
    xts = [xt_.clone().detach().cuda(0), xt_.clone().detach().cuda(1)]
    cInvs = [currentInv.clone().detach().cuda(0), currentInv.clone().detach().cuda(1)]
    # xts = [xt_[:total_len//2].clone().detach().cuda(0),
    #        xt_[total_len//2:].clone().detach().cuda(1)]
    mid = time.time()
    print('Random arrays sent to gpus in', mid - start)

    # Moved Pool creation outside and pushed the for-loop inside 
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(30)
    NUM_GPUS = torch.cuda.device_count()
    traceEst = np.frombuffer(tE.get_obj())
    print("Before pool", traceEst.shape)
    here = time.time()
    with Pool(processes=NUM_GPUS, initializer=initpool, initargs=(tE,)) as pool:
        # args = [(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
        args = [(xts[x], rank, chunkSize, NUM_GPUS, cInvs[x], fishers[x], x) for x in range(NUM_GPUS)]
        for i in range(5):
            result = pool.starmap(trace_for_chunk, args)
            # pool.close()
            # pool.join()
    print('With took', time.time()- here)

    # Using queue to share memory (giving errors)
    # manager = Manager()
    # q = manager.Queue()
    # processes = []
    # for _ in range(NUM_GPUS):
    #     gpu_id = [x for x in range(NUM_GPUS)]
    #     args = [(xts[x], rank, chunkSize, NUM_GPUS, cInvs[x], fishers[x],q, x) for x in range(NUM_GPUS)]
    #     p  = mp.Process(target=trace_for_chunk(xt_, rank, chunkSize, NUM_GPUS,currentInv,fisher, q, gpu_id),args=args)
    #     p.start()
    #     processes.append(p)

    # print(q.get(), time.time())







