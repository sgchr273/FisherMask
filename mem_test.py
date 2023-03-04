import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
import resnet
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import gc
import time
import nvsmi

data_tr = datasets.CIFAR10('data'+ '/CIFAR10', train=True, download=True)
data_te = datasets.CIFAR10('data' + '/CIFAR10', train=False, download=True)
X_tr = data_tr.data
Y_tr = torch.from_numpy(np.array(data_tr.targets))
X_te = data_te.data
Y_te = torch.from_numpy(np.array(data_te.targets))

num_imp_per_layer = 7000
xt = np.zeros((len(Y_tr), len(np.unique(Y_tr)), num_imp_per_layer))

#arguments for select
fisher = torch.zeros(xt.shape[-1], xt.shape[-1],dtype=torch.double).cuda()
iterates = torch.zeros(xt.shape[-1], xt.shape[-1], dtype=torch.double)
n_pool = len(Y_tr)
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_unlabeled = np.arange(n_pool)[~idxs_lb]

def select(X, K, fisher, iterates, lamb=1, backwardSteps=0, nLabeled=0):
    '''
    K is the number of images to be selected for labelling, 
    iterates is the fisher for images that are already labelled
    '''
    # print(X.shape, fisher.shape, iterates.shape)
    numEmbs = len(X)
    dim = X.shape[-1]
    rank = X.shape[-2]
    indsAll = []

    start_select = time.time()
    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    # what is lamb used for here?
    #X = X * np.sqrt(K / (nLabeled + K))
    inv_time = time.time()
    # print("inverse op took ", inv_time - start_select)
    fisher = fisher.cuda()
    # print("placing fisher in cuda", time.time() - inv_time)
    total = 0
    total_outer = 0
    # forward selection
    for i in range(int((backwardSteps + 1) *  K)):
        if i % 10 == 0:
            print(i)
            print(nvsmi.get_gpus())
            print(nvsmi.get_available_gpus())
            print(nvsmi.get_gpu_processes())
        xt_ = X 


        traceEst = np.zeros(X.shape[0]) #torch.zeros(X.shape[0]).cuda() 
        chunkSize = 50
        #print(X.shape[0])
        
        time_for_inner_loop = time.time()
        for c_idx in range(0, X.shape[0], chunkSize):
            xt_chunk = xt_[c_idx * chunkSize : (c_idx + 1) * chunkSize]
            xt_chunk = torch.tensor(xt_chunk).cuda() #.clone().detach()
            innerInv = torch.inverse(torch.eye(rank).cuda() + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2))
            innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
            traceEst[c_idx * chunkSize : (c_idx + 1) * chunkSize] = torch.diagonal(
                xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv, 
                dim1=-2, 
                dim2=-1
            ).sum(-1).detach().cpu()
        time_for_inner_loop_end = time.time()
        #print('inner loop time: ', time_for_inner_loop_end-time_for_inner_loop)
        total += (time_for_inner_loop_end-time_for_inner_loop)

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
       
        xt_ = torch.tensor(X[ind]).unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]
        time_for_outer_loop = time.time()
        total_outer += time_for_outer_loop - time_for_inner_loop_end

    # backward pruning
    second_for_loop_time = time.time()
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
    second_for_loop_time_end = time.time()
    # print("The second for loop in the select function took ", (second_for_loop_time_end-second_for_loop_time))
    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    # print("final part of select takes", time.time()-second_for_loop_time_end)
    return indsAll


chosen = select(xt[idxs_unlabeled], 1000, fisher, iterates, lamb = 1, backwardSteps = 0, nLabeled=np.sum(idxs_lb))