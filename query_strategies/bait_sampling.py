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

def select(X, K, fisher, iterates, lamb=1, backwardSteps=0, nLabeled=0):
    '''
    K is the number of images to be selected for labelling, 
    iterates is the fisher for images that are already labelled
    '''

    numEmbs = len(X)
    dim = X.shape[-1]
    rank = X.shape[-2]
    indsAll = []

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    # what is lamb used for here?
    #X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()
    total = 0
    # forward selection
    for i in range(int((backwardSteps + 1) *  K)):
        print("Select function for loop: ", i)
        '''
        K corresponds to minibatch size, which is called B in the paper.
        currently we assume that backwardSteps = 0
        '''

        # xt_ = X.cuda()
        xt_ = X  
        '''
        in the calculation below, traceEst has X.shape[0] elements.
        The calculation done for computing one element of traceEst
        has no effect on the calculation done for computing other
        elements of traceEst. This suggests that we can compute  
        tracEst in chunks, rather than computing all elements in 
        one go.

        traceEst = torch.zeros(X.shape[0])
        chunkSize = 100
        for c_idx in range(0, X.shape[0], chunkSize):
            xt_chunk = xt_[c_idx * chunkSize : (c_idx + 1) * chunkSize]
            innerInv = torch.inverse(torch.eye(rank).cpu() + xt_chunk @ currentInv @ xt_chunk.transpose(1, 2)).detach()
            traceEst[c_idx * chunkSize : (c_idx + 1) * chunkSize] = torch.diagonal(
                xt_chunk @ currentInv @ fisher @ currentInv @ xt_chunk.transpose(1, 2) @ innerInv, 
                dim1=-2, 
                dim2=-1
            ).sum(-1) 
        '''


        # innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        # innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        
        
        
        # traceEst = torch.diagonal(
        #     xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, 
        #     dim1=-2, 
        #     dim2=-1
        # ).sum(-1)
        traceEst = np.zeros(X.shape[0]) #torch.zeros(X.shape[0]).cuda() 
        chunkSize = 100
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
        '''
        Vx^T M^-1 I(θ_L) M^-1 Vx A^-1 formula from page 5 of paper.
        currentInv corresponds to M^-1
        fisher corresponds to I(θ_L)
        xt_ corresponds to Vx^T
        innerInv corresponds to A^-1
        '''

        xt = xt_
        del xt, innerInv
        #del xt_, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

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
       
        xt_ = torch.tensor(X[ind]).unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    print("Average time for inner for loop of select function: ", (total/int((backwardSteps + 1) *  K)))
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
    print("The second for loop in the select function took ", (second_for_loop_time_end-second_for_loop_time))
    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll

class BaitSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

        self.fishIdentity = args['fishIdentity']
        self.fishInit = args['fishInit']
        self.lamb = args['lamb']
        self.backwardSteps = args['backwardSteps']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
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

        # get fisher
        if self.fishIdentity == 0:
            print('getting fisher matrix ...', flush=True)
            batchSize = 1000
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
            rounds = int(np.ceil(len(self.X) / batchSize))
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
        batchSize = 1000 
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
        
        chosen = select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb))
        print('selected probs: ' +
                str(str(torch.mean(torch.max(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.min(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.std(phat[chosen,:], 1)).item())), flush=True)
        return idxs_unlabeled[chosen]
