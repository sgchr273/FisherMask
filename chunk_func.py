import torch
import time
import numpy as np
import matplotlib.pyplot as plt

xt_ = torch.rand(1000, 10, 15000)#.numpy()
rank = xt_.shape[-2]
random_mat = torch.rand(xt_.shape[-1], xt_.shape[-1])
random_mat2 = torch.rand(xt_.shape[-1], xt_.shape[-1])

currentInv = torch.inverse(random_mat @ random_mat.T).cuda()
#cInv = currentInv.detach().cpu().numpy()
fisher = (random_mat2 @ random_mat2.T).cuda()
#fish = fisher.detach().cpu().numpy()

def time_chunk(chunkSize, num_GPUS, streams):
    total=0
    traceEst = np.zeros(xt_.shape[0])
    chunkSize = min(xt_.shape[0], chunkSize)
    time_for_inner_loop = time.time()
    for i, c_idx in enumerate(range(0, xt_.shape[0], chunkSize)):
        with torch.cuda.stream(streams[i%num_GPUS]):
            xt_chunk = xt_[c_idx : c_idx + chunkSize]
            xt_chunk = xt_chunk.clone().detach().cuda(i%num_GPUS) #torch.tensor()
            innerInv = torch.inverse(torch.eye(rank).cuda(i%num_GPUS) + xt_chunk @ currentInv.cuda(i%num_GPUS) @ xt_chunk.transpose(1, 2))
            traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
                xt_chunk @ currentInv.cuda(i%num_GPUS) @ fisher.cuda(i%num_GPUS) @ currentInv.cuda(i%num_GPUS) @ xt_chunk.transpose(1, 2) @ innerInv, 
                dim1=-2, 
                dim2=-1
            ).sum(-1).detach().cpu()
    time_for_inner_loop_end = time.time()
    return time_for_inner_loop_end-time_for_inner_loop
    #return total



def plot_chunk_exp(cSizes, savefile = 'chunk_1_500_10'):
    stream = [torch.cuda.Stream(x) for x in range(1)]
    times = [time_chunk(x, 1, stream) for x in cSizes]
    plt.plot(cSizes, times, '-o')
    plt.savefig("chunksize_exp"+savefile+".png")
    plt.close()
    return cSizes[np.argsort(times)]


chunkSizes = np.arange(1,100,5)
plot_chunk_exp(chunkSizes)
