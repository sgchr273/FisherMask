import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Array, Queue, Manager
import numpy as np
import time
import gc
from scipy import stats


def initpool(arr):
    global sharedArr
    sharedArr = arr

def trace_for_chunk(xt_, rank, num_gpus, chunkSize, currentInv,fisher, total_len, gpu_id):
    # total_len = xt_.shape[0] #* num_gpus
    # upper_bound = int(total_len/(num_gpus-gpu_id))
    # lower_bound = int(upper_bound-(total_len/num_gpus))
    # traceEst = np.frombuffer(sharedArr.get_obj(), dtype=np.float32)

    # print("Inside trace_for_chunk",  chunkSize)
    
    print("Beginning GPU ", gpu_id, " at time: ", time.ctime(), flush=True)
    # xt_chunk = xt_[lower_bound : upper_bound]
    # queue.put(xt_chunk)
    # queue.put(fisher)
    # queue.put(currentInv)
    print(len(xt_), chunkSize)
    traceEst = torch.zeros((total_len//num_gpus))
    for c_idx in range(len(xt_), chunkSize):
        xt_chunk = xt_[c_idx : c_idx + chunkSize]
        xt_chunk = xt_chunk.cuda(gpu_id)
        fisher = fisher.cuda(gpu_id)
        currentInv = currentInv.cuda(gpu_id)
        # print(F"Sent to {gpu_id} in", time.time() - send_time)
        # with torch.no_grad():
        innerInv_time = time.time()
        innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2)) 
        
        # print("InnerInv at GPU", innerInv.get_device(), " took ", time.time()-innerInv_time, "has shape", innerInv.shape)
        tracest_time = time.time()
        traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
            xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1
        ).sum(-1).detach().cpu()
        # print("traceEst calculated in", time.time() - tracest_time)
   
    
    # print("Finishing GPU ", gpu_id, " at time: ", time.ctime(), flush=True)
    return traceEst

def betterSlice(num_gpus, gpu_id, total_len):
    upper_bound = int(total_len/(num_gpus-gpu_id))
    lower_bound = int(upper_bound-(total_len/num_gpus))
    return slice(lower_bound, upper_bound)
    
if __name__ == '__main__':
    
    here1 = time.time()
    rng = np.random.default_rng()
    dim = 10 * 700 #X.shape[-1]
    rank = 10 #X.shape[-2]
    fisher = torch.rand((dim, dim))
    currentInv = torch.rand((dim, dim))
    X = torch.rand((30000, rank, dim)) #rng.random((2000, rank, dim))
    total_len = X.shape[0]
    chunkSize = 500 #len(xt_) // 4 #
    # tE = Array('d', total_len, lock=True)
    print('Initialization: ', time.time() - here1)
    


    start = time.time()
    NUM_GPUS = torch.cuda.device_count()
    # NUM_GPUS = 1
    fishers = [fisher.clone().detach().cuda(0), fisher.clone().detach().cuda(1)]
    xts = [X[betterSlice(NUM_GPUS, 0, total_len)].clone().detach().cuda(0), X[betterSlice(NUM_GPUS, 1, total_len)].clone().detach().cuda(1)]
    # xts = [X[betterSlice(NUM_GPUS, 0, total_len)].clone().detach().cuda(0)]
    cInvs = [currentInv.clone().detach().cuda(0), currentInv.clone().detach().cuda(1)]
    # xts = [xt_[:total_len//2].clone().detach().cuda(0),
    #        xt_[total_len//2:].clone().detach().cuda(1)]
    mid = time.time()
    print('Random arrays sent to gpus in', mid - start)

    # Moved Pool creation outside and pushed the for-loop inside 
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(30)
    
    # NUM_GPUS  = 1
    
    # print("Before pool", traceEst.shape)
    here = time.time()
    indsAll = []
    with Pool(processes=NUM_GPUS) as pool:
        # args = [(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
        # args = [(xts[x], rank, chunkSize, NUM_GPUS, cInvs[x], fishers[x], x) for x in range(NUM_GPUS)]
        for i in range(5):
            xts[0] = X[betterSlice(NUM_GPUS, 0, total_len)].clone().detach().cuda(0)
            args = [(xts[x], rank, NUM_GPUS, chunkSize, cInvs[x], fishers[x], total_len, x) for x in range(NUM_GPUS)]
            tE = pool.starmap(trace_for_chunk, args)
            traceEst = tE[0]
            for j in range(1,NUM_GPUS):
                traceEst = torch.cat((traceEst, tE[j]))
            # xt = xt_.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            print(len(traceEst))
            traceEst = traceEst.detach().cpu().numpy()

            dist = traceEst - np.min(traceEst) + 1e-10
            dist = dist / np.sum(dist)
            sampler = stats.rv_discrete(values=(np.arange(len(dist)), dist))
            ind = sampler.rvs(size=1)[0]
            for j in np.argsort(dist)[::-1]:
                if j not in indsAll:
                    ind = j
                    break

            indsAll.append(ind)
            xts[0] = X[ind].unsqueeze(0).cuda()
            innerInv = torch.inverse(torch.eye(rank).cuda(0) + xts[0] @ cInvs[0] @ xts[0].transpose(1, 2)).detach()
            cInvs[0] = (cInvs[0] - cInvs[0] @ xts[0].transpose(1, 2) @ innerInv @ xts[0] @ cInvs[0]).detach()[0]
            cInvs[1] = cInvs[0]
            # pool.close()
            # pool.join()
    # traceEst = np.frombuffer(tE.get_obj())
    print("After pool", traceEst.shape)
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







