import numpy as np
import torch
from .strategy import Strategy
from saving import save_queried_idx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import time
from .strategy import Strategy
import gc
import sys
import pickle
import os
import logging
import random
from .bait_sampling import select, fresh_select
from saving import save_imp_weights, save_queried_idx
from torch.nn.parallel import DataParallel
from fisher_mask_sampling import calculate_fishmask, log_prob_grads_wrt

class FishEntSampling(Strategy):
      
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(FishEntSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.savefile = args["savefile"]
        self.fishIdentity = args['fishIdentity']
        self.fishInit = args['fishInit']
        self.lamb = args['lamb']
        self.backwardSteps = args['backwardSteps']
        self.pct_top = args['pct_top']
        self.savefile = args["savefile"]
        self.chunkSize = args["chunkSize"]

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        chosen_ent = idxs_unlabeled[U.sort()[1][:n/2]]
        remaining_points = idxs_unlabeled[n/2:]
        
        self.fishIdentity == 0
        imp_wt_idxs = self.calculate_fishmask(self.pct_top, method ='standard') 
        save_imp_weights(imp_wt_idxs, self.savefile)
        start = time.time()
        xt = self.log_prob_grads_wrt(imp_wt_idxs)
        print('log_prob_grads took:', time.time()- start)
        torch.cuda.empty_cache()
        gc.collect()

        batchSize = 17
        # get fisher
        if self.fishIdentity == 0:
            print('getting fisher matrix ...', flush=True)
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
            for i in range(int(np.ceil(len(self.X) / batchSize))):
                '''
                adding individual fisher matrices to compute overall fisher matrix I(theta_^L_t)
                '''
                xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
                fisher = fisher + op
                del xt_, op
                torch.cuda.empty_cache()
                gc.collect()
        else: fisher = torch.eye(xt.shape[-1])
        
        # get fisher only for samples that have been seen before
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        if self.fishInit == 1:
            sec_time = time.time()
            for i in range(int(np.ceil(len(xt2) / batchSize))):
                
                # xt_ = torch.tensor(xt2[i * batchSize : (i + 1) * batchSize]).cuda()
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
        
        # xt = xt[idxs_unlabeled]
        # xt = xt.double()
        sel1 = time.time()
        chosen_fish = select(xt[remaining_points], n/2, fisher, init, self.savefile, "FISH", lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb), chunkSize=self.chunkSize)
        # chosen = fresh_select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb))

        print('Select took:', time.time()-sel1)
        print('selected probs: ' +
                str(str(torch.mean(torch.max(phat[chosen_fish, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.min(phat[chosen_fish, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.std(phat[chosen_fish,:], 1)).item())), flush=True)
        chosen = np.concatenate([chosen_ent, chosen_fish])
        save_queried_idx(idxs_unlabeled[chosen], self.savefile, self.alg)
        return idxs_unlabeled[chosen]
