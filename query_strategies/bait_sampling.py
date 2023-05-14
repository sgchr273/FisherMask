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

def initpool(arr):
    global sharedArr
    sharedArr = arr


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


def save_queried_idx(idx,filename):
    try:
        savefile = open("./Save/Queried_idxs/bait_queried_idxs_"+ filename+'.p', "br")
        que_idxs = pickle.load(savefile)
        savefile.close()
    except:
        que_idxs = []
    finally:
        if not os.path.exists("./Save/Queried_idxs"):
            os.makedirs("./Save/Queried_idxs")
        savefile = open("./Save/Queried_idxs/bait_queried_idxs_"+ filename+'.p', "bw")
        que_idxs.append(idx)
        pickle.dump(que_idxs, savefile)
        savefile.close()

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


def select(X, K, fisher, iterates, lamb=1, backwardSteps=0, nLabeled=0, chunkSize=200):
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
    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
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

        here1 = time.time()

        with Pool(processes=NUM_GPUS, initializer=initpool, initargs=(tE,)) as pool:
            args = [(xt_, rank, chunkSize, NUM_GPUS, currentInv, fisher, x) for x in range(NUM_GPUS)]
            result = pool.starmap(trace_for_chunk, args)
        here2 = time.time()
        print(here2 - here1)


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
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]
        #time_for_outer_loop = time.time()
        #total_outer += time_for_outer_loop - time_for_inner_loop_end
        # print('xt: ', xt_ ,'\n', 'currentinv: ', currentInv, '\n', 'innerinv: ',  innerInv,'\n', 'fisher: ', fisher)

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

class BaitSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

        self.fishIdentity = args['fishIdentity']
        self.fishInit = args['fishInit']
        self.lamb = args['lamb']
        self.backwardSteps = args['backwardSteps']
        self.savefile = args["savefile"]
        self.chunkSize = args["chunkSize"]

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
            batchSize = 500
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
        batchSize = 500
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
        start_time = time.time()
        chosen = select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb), chunkSize=self.chunkSize)
        end_time = time.time()
        print('Time taken by select using 2 gpus:', end_time - start_time)
        save_queried_idx(idxs_unlabeled[chosen], self.savefile)
        print('selected probs: ' +
                str(str(torch.mean(torch.max(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.min(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.std(phat[chosen,:], 1)).item())), flush=True)
        return idxs_unlabeled[chosen]
