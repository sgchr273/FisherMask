import torch
import time
import gc
import numpy as np
from scipy import stats
from torch.multiprocessing import Pool, Queue, Manager, Array
from torchvision import datasets
from torch.utils.data import Dataset


def initpool(arr):
    global sharedArr
    sharedArr = arr

def trace_for_chunk(xt_, rank, chunkSize, num_gpus, currentInv, fisher, gpu_id):
    upper_bound = int(xt_.shape[0]/(num_gpus-gpu_id))
    lower_bound = int(upper_bound-(xt_.shape[0]/num_gpus))
    here1 = time.time()
    # traceEst = np.frombuffer(sharedArr.get_obj(), dtype=np.float64)
    traceEst = np.frombuffer(sharedArr.get_obj(), dtype=np.double)
    here2 = time.time()
    print('Trace est time =', here2 - here1)
    # print("Beginning GPU ", gpu_id, " at time: ", time.time(), flush=True)
    for c_idx in range(lower_bound, upper_bound, chunkSize):
        xt_chunk = xt_[c_idx : c_idx + chunkSize]
        # xt_chunk = torch.tensor(xt_chunk).clone().detach().cuda(gpu_id)
        xt_chunk = xt_chunk.clone().detach().cuda(gpu_id)
        currentInv = currentInv.cuda(gpu_id)
        fisher = fisher.cuda(gpu_id)
        innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
        # innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]).max
        traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
            xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1
        ).sum(-1).detach().cpu()
    del xt_chunk, fisher, currentInv
    here3 = time.time()
    print('Trace_for_chunk for loop took', here3 - here2)
    # print("Finishing GPU ", gpu_id, " at time: ", time.time(), flush=True)
    return


def select(X, K, fisher, iterates, lamb=1, backwardSteps=0, nLabeled=0, chunkSize=500):
    '''
    K is the number of images to be selected for labelling, 
    iterates is the fisher for images that are already labelled
    '''
    # print(X.shape, fisher.shape, iterates.shape)
    numEmbs = len(X)
    dim = X.shape[-1]
    rank = X.shape[-2]
    indsAll = []

    #start_select = time.time()
    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K)).float()
    # what is lamb used for here?
    X = X * np.sqrt(K / (nLabeled + K))
    #inv_time = time.time()
    # print("inverse op took ", inv_time - start_select)
    fisher = fisher.cuda()
    # print("placing fisher in cuda", time.time() - inv_time)
    total = 0
    #total_outer = 0
    # forward selection
    for i in range(int((backwardSteps + 1) *  K)):
        # print("Select function for loop: ", i)
        '''
        K corresponds to minibatch size, which is called B in the paper.
        currently we assume that backwardSteps = 0
        '''

        # xt_ = X.cuda()
        xt_ = X  

        # innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        # innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        
        
        
        # traceEst = torch.diagonal(
        #     xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, 
        #     dim1=-2, 
        #     dim2=-1
        # ).sum(-1)
        #traceEst = np.zeros(X.shape[0]) #torch.zeros(X.shape[0]).cuda() 
        chunkSize = min(X.shape[0], chunkSize) # replace 100 by chunkSize argument
        
        time_for_inner_loop = time.time()
        """ for c_idx in range(0, X.shape[0], chunkSize):
            xt_chunk = xt_[c_idx : c_idx + chunkSize]
            xt_chunk = torch.tensor(xt_chunk).cuda() #.clone().detach()
            innerInv = torch.inverse(torch.eye(rank).cuda() + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
            innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
            traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
                xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv, 
                dim1=-2, 
                dim2=-1
            ).sum(-1).detach().cpu() """
        NUM_GPUS = torch.cuda.device_count()
        # logging.debug("Inside select funtion for loop" + str(NUM_GPUS))
        torch.multiprocessing.set_start_method('spawn', force=True)
        tE = Array('d', xt_.shape[0], lock=True)
        traceEst = np.frombuffer(tE.get_obj())

        here4 = time.time()


        with Pool(processes=NUM_GPUS, initializer=initpool, initargs=(tE,)) as pool:
            args = [(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
            result = pool.starmap(trace_for_chunk, args)
        
        here5 = time.time()
        print('With took ', here5 - here4)


        time_for_inner_loop_end = time.time()
        #print('inner loop time: ', time_for_inner_loop_end-time_for_inner_loop)
        total += (time_for_inner_loop_end-time_for_inner_loop)
        '''
        def time_chunk(chunkSize):
            total=0
            traceEst = np.zeros(xt_.shape[0])
            chunkSize = min(xt_.shape[0], chunkSize)
            time_for_inner_loop = time.time()
            for c_idx in range(0, X.shape[0], chunkSize):
                xt_chunk = xt_[c_idx : c_idx + chunkSize]
                xt_chunk = xt_chunk.clone().detach().cuda() #torch.tensor()
                innerInv = torch.inverse(torch.eye(rank).cuda() + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
                traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
                    xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv, 
                    dim1=-2, 
                    dim2=-1
                ).sum(-1).detach().cpu()
            time_for_inner_loop_end = time.time()
            total += (time_for_inner_loop_end-time_for_inner_loop)
            return total

        Vx^T M^-1 I(θ_L) M^-1 Vx A^-1 formula from page 5 of paper.
        currentInv corresponds to M^-1
        fisher corresponds to I(θ_L)
        xt_ corresponds to Vx^T
        innerInv corresponds to A^-1
        '''

        # xt = xt_
        # del xt, innerInv
        #del xt_, innerInv
        # torch.cuda.empty_cache()
        # gc.collect()
        # torch.cuda.empty_cache()
        # gc.collect()

        # traceEst = traceEst.detach().cpu().numpy() # objective value in eq (5) from the paper

        dist = traceEst - np.min(traceEst) + 1e-10
        dist = dist / np.sum(dist)
        sampler = stats.rv_discrete(values=(np.arange(len(dist)), dist))
        ind = sampler.rvs(size=1)[0]
        for j in np.argsort(dist)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)  # adding a new tilde_x to the minibatch being made
        #print(i, ind, traceEst[ind], flush=True)
       
        # xt_ = torch.tensor(X[ind]).unsqueeze(0).cuda()
        xt_ = (X[ind]).unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ .float()@ currentInv @ xt_.float().transpose(1, 2)).detach().float()
        currentInv = (currentInv - currentInv @ xt_.float().transpose(1, 2) @ innerInv @ xt_.float()@ currentInv).detach()[0]
        # print(type(innerInv), ' ',  type(currentInv), ' ', type(xt_), type(rank))
        #time_for_outer_loop = time.time()
        #total_outer += time_for_outer_loop - time_for_inner_loop_end

    # logging.debug("Average time of chunk loop: " + str(total/int((backwardSteps + 1) *  K)))
    # print("Average time for outer for loop of select function: ", (total_outer/int((backwardSteps + 1) *  K)))
    # backward pruning
    #second_for_loop_time = time.time()
    rounds = len(indsAll) - K
    for i in range(rounds):

        # select index for removal
        xt_ = torch.tensor(X[indsAll]).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        #print(i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


        # compute new inverse
        xt_ = torch.tensor(X[indsAll[delInd]]).unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]
    #second_for_loop_time_end = time.time()
    # print("The second for loop in the select function took ", (second_for_loop_time_end-second_for_loop_time))
    # del xt_, innerInv, currentInv, tE, traceEst, sharedArr
    del xt_, innerInv, currentInv, tE, traceEst
    torch.cuda.empty_cache()
    gc.collect()
    # print("final part of select takes", time.time()-second_for_loop_time_end)
    return indsAll

def get_dataset(name, path):
    if name == 'CIFAR10':
        return get_CIFAR10(path)

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

X_tr, Y_tr, X_te, Y_te = get_dataset('CIFAR10','data')
n_pool = len(Y_tr)
embDim =50
nLab = len(np.unique(Y_tr))
idxs_lb = np.zeros(n_pool, dtype=bool)
embedding = np.zeros([len(Y_tr), nLab, embDim * nLab])
xt = torch.tensor(embedding)
idxs_unlabeled = np.arange(n_pool)[~idxs_lb]
fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
init = torch.zeros(xt.shape[-1], xt.shape[-1])

n = 2000
if __name__ == '__main__':
   select(xt[idxs_unlabeled], n, fisher, init, lamb=1, backwardSteps=0, nLabeled=np.sum(idxs_lb), chunkSize=200)