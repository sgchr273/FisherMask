import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
import gc
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
# import torchfile
from torch.autograd import Variable
import resnet
import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import numpy as np
import scipy.sparse as sp
from itertools import product
import os
import time
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
# from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.exceptions import ConvergenceWarning
# from sklearn.metrics import pairwise_distances
import logging
from torch.multiprocessing import Pool, Queue, Manager, Array
import subprocess
from saving import save_queried_idx, save_dist_stats

def betterSlice(num_gpus, gpu_id, total_len):
    upper_bound = (1+gpu_id)*total_len/num_gpus
    lower_bound = upper_bound-(total_len/num_gpus)
    return slice(int(lower_bound), int(upper_bound))


# kmeans ++ initialization
def batchOuterProdDet(X, A, batchSize):
    dets = []
    rank = X.shape[-2]
    batches = int(np.ceil(len(X) / batchSize))
    for i in range(batches):
        x = X[i * batchSize : (i + 1) * batchSize].cuda()
        outerProds = (torch.matmul(torch.matmul(x, A), torch.transpose(x, 1, 2))).detach()
        newDets = (torch.det(outerProds + torch.eye(rank).cuda())).detach()
        dets.append(newDets.cpu().numpy())

    dets = np.abs(np.concatenate(dets))
    dets[np.isinf(dets)] = np.finfo('float32').max
    dist = dets / np.finfo('float32').max
    dist -= np.min(dist)
    dist /= np.sum(dist)
    return dist, dets


def getUse():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def trace_for_chunk(xt_, rank, num_gpus, chunkSize, currentInv, fisher, total_len, gpu_id):
    chunkSize = min(xt_.shape[0], chunkSize)
    traceEst = torch.zeros((len(xt_)))
    for c_idx in range(0, len(xt_), chunkSize):
        xt_chunk = xt_[c_idx : c_idx + chunkSize]
        xt_chunk = xt_chunk.cuda(gpu_id)
        fisher = fisher.cuda(gpu_id)
        currentInv = currentInv.cuda(gpu_id)
        # with torch.no_grad():
        innerInv = torch.inverse(torch.eye(rank).cuda(gpu_id) + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2)) 
        traceEst[c_idx : c_idx + chunkSize] = torch.diagonal(
            xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1
        ).sum(-1).detach().cpu()
    return traceEst


def fresh_select(X, K, fisher, iterates, lamb=1, backwardSteps=0, nLabeled=0):

    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    # forward selection
    for i in range(int((backwardSteps + 1) *  K)):

        xt_ = X.cuda() 
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

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
        # print(i, ind, traceEst[ind], flush=True)
       
        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    for i in range(len(indsAll) - K):

        # select index for removal
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        # print(i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


        # compute new inverse
        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll




def select(X, K, fisher, iterates, savefile, alg, lamb=1, backwardSteps=0, nLabeled=0, chunkSize=200):
    '''
    K is the number of images to be selected for labelling, 
    iterates is the fisher for images that are already labelled
    '''
    time_begin_select = time.time()
    numEmbs = len(X)
    dim = X.shape[-1]
    rank = X.shape[-2]
    indsAll = []
    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    xt_ = X
    #chunkSize = min(X.shape[0], chunkSize)
    total_len = xt_.shape[0]
    NUM_GPUS = torch.cuda.device_count()
    fishers = [fisher.clone().detach().cuda(x) for x in range(NUM_GPUS)]
    xts = [X[betterSlice(NUM_GPUS, x, total_len)].clone().detach().cuda(x) for x in range(NUM_GPUS)]
    torch.multiprocessing.set_start_method('spawn', force=True)
    distStats = []

        
    with Pool(processes=NUM_GPUS) as pool:
        for i in range(int((backwardSteps + 1) *  K)):
            cInvs = [currentInv.clone().detach().cuda(x) for x in range(NUM_GPUS)]
            args = [(xts[x], rank, NUM_GPUS, chunkSize, cInvs[x], fishers[x], total_len, x) for x in range(NUM_GPUS)]
            tE = pool.starmap(trace_for_chunk, args)
            traceEst = tE[0]
            for j in range(1,NUM_GPUS):
                traceEst = torch.cat((traceEst, tE[j]))
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            traceEst = traceEst.detach().cpu().numpy()

            dist = traceEst - np.min(traceEst) + 1e-10
            dist = dist / np.sum(dist)
            distStats.append([np.min(dist), np.max(dist), np.std(dist)])
            sampler = stats.rv_discrete(values=(np.arange(len(dist)), dist))
            ind = sampler.rvs(size=1)[0]
            for j in np.argsort(dist)[::-1]:
                if j not in indsAll:
                    ind = j
                    break

            indsAll.append(ind)
            temp_xt = X[ind].unsqueeze(0).cuda()
            innerInv = torch.inverse(torch.eye(rank).cuda(0) + temp_xt @ cInvs[0] @ temp_xt.transpose(1, 2)).detach()
            currentInv = (cInvs[0] - cInvs[0] @ temp_xt.transpose(1, 2) @ innerInv @ temp_xt @ cInvs[0]).detach()[0]

    for i in range(len(indsAll) - K):
        # select index for removal
        xt_ = torch.tensor(X[indsAll]).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()

        # compute new inverse
        xt_ = torch.tensor(X[indsAll[delInd]]).unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv, tE, traceEst
    save_dist_stats(distStats, savefile, alg)
    torch.cuda.empty_cache()
    gc.collect()
    time_end_select = time.time()
    logging.debug("Select took" + str(time_end_select - time_begin_select) + "seconds")
    return indsAll

class BaitSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.alg = "BAIT"
        self.fishIdentity = args['fishIdentity']
        self.fishInit = args['fishInit']
        self.lamb = args['lamb']
        self.backwardSteps = args['backwardSteps']
        self.savefile = args["savefile"]
        self.chunkSize = args["chunkSize"]

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        print(len(idxs_unlabeled))
        '''
        idxs_lb stands for indexes_labeled, i.e. image idxs that have been labeled.
        idxs_lb is a 0-1 vector having length as n_pool, a particular component is 
        0 if the corresponding image has not been labeled previously, otherwise it is 1.
        ~idxs_lb is the componentwise NOT operation on idxs_lb, i.e., components with 1
        are images that have not been labeled previously.
        n_pool is size of entire training dataet, i.e. in CIFAR10, it is 60,000.
        idxs_unlabeled is a np array of integer indices for images that have not been 
        labeled previously.
        '''

        # get low rank fishers
        xt = self.get_exp_grad_embedding(self.X, self.Y)
        '''
        X has all the training images and Y has all the corresponding labels.
        get_exp_grad_embedding corresponds to Appendix A.2, 
        and is calculating the Vx matrix.
        For us, xt should contain the gradients wrt all the important weights.
        '''
        batchSize = 250
        # get fisher
        if self.fishIdentity == 0:
            print('getting fisher matrix ...', flush=True)
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
            rounds = int(np.ceil(len(self.X) / batchSize))
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            for i in range(int(np.ceil(len(self.X) / batchSize))):
                '''
                adding individual fisher matrices to compute overall fisher matrix I_U
                '''
                xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
                fisher = fisher + op
                xt_ = xt_.cpu()
                del xt_, op
                torch.cuda.empty_cache()
                gc.collect()
        else: fisher = torch.eye(xt.shape[-1])


        # get fisher only for samples that have been seen before
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        rounds = int(np.ceil(len(xt2) / batchSize))
        if self.fishInit == 1:
            for i in range(int(np.ceil(len(xt2) / batchSize))):
                xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
                init = init + op
                xt_ = xt_.cpu()
                del xt_, op
                torch.cuda.empty_cache()
                gc.collect()

        phat = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        print('all probs: ' + 
                str(str(torch.mean(torch.max(phat, 1)[0]).item())) + ' ' + 
                str(str(torch.mean(torch.min(phat, 1)[0]).item())) + ' ' + 
                str(str(torch.mean(torch.std(phat,1)).item())), flush=True)
        
        chosen = fresh_select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb))
        save_queried_idx(idxs_unlabeled[chosen], self.savefile, self.alg)

        print('selected probs: ' +
                str(str(torch.mean(torch.max(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.min(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.std(phat[chosen,:], 1)).item())), flush=True)
        
        return idxs_unlabeled[chosen]
