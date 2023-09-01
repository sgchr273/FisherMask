import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
# from torchvision.models import ResNet18_Weights
import time
from .strategy import Strategy
import gc
import sys
import pickle
import os
import logging
import random
from .bait_sampling import select
from saving import save_imp_weights, save_queried_idx

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class fisher_mask_sampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, method):
        super(fisher_mask_sampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.alg = "FISH"
        self.fishIdentity = args['fishIdentity']
        self.fishInit = args['fishInit']
        self.lamb = args['lamb']
        self.backwardSteps = args['backwardSteps']
        self.pct_top = args['pct_top']
        self.savefile = args["savefile"]
        self.chunkSize = args["chunkSize"]
        self.method = method
        self.rand_mask = self.calculate_random_mask(1280)

    def calculate_fishmask(self, pct_top=0.02, method="standard"):
        #---------------------------------Originally calculate_gradients---------------------------------
        self.net.to(device)
        for param in self.net.parameters():
            param.requires_grad = True
        self.net.eval()
        parameters = tuple(self.net.parameters())
        sq_grads_expect = {i: np.zeros(p.shape) for i, p in enumerate(parameters)}
        test_loader = DataLoader(self.handler(self.X, self.Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        num_samples = 1024 # used by FISH mask paper
        idx = 0
        for test_batch, test_labels, idxs in test_loader:
            if idx >= num_samples:
                break
            
            test_batch, test_labels = test_batch.cuda(), test_labels.cuda()
            outputs, e1 = self.net(test_batch)
            _, preds = torch.max(outputs, 1)         
            probs = F.softmax(outputs, dim=1).to('cpu')
            log_probs = F.log_softmax(outputs, dim=1)
            N, C = log_probs.shape

            for n in range(N):
                for c in range(C):
                    grad_list = torch.autograd.grad(log_probs[n][c], parameters, retain_graph=True)
                    for i, grad in enumerate(grad_list):    # different layers
                        gsq = torch.square(grad).to('cpu') * probs[n][c] / N
                        sq_grads_expect[i] += gsq.detach().numpy() # sq_grads_expect[i] + gsq
                        del gsq
                    self.net.zero_grad()
                idx += 1
                if idx >= num_samples:
                    break

        #---------------------------------Originally calculate_mask---------------------------------
        if method == "standard":
            list_t = list(sq_grads_expect.values()) # same dim as model
            combined_arrays = np.hstack([t.flatten() for t in sq_grads_expect.values()]) # dim 61 with grad values flattened for each layer
            list_lengths = [len(ten.flatten()) for ten in list_t] # dim 61 with size of each layer
            cum_lengths = np.cumsum(list_lengths)
            sorted_idxs = np.argsort(combined_arrays[:cum_lengths[-2]])
            num_top = int(pct_top * len(combined_arrays))
            # top_idxs = sorted_idxs[-num_top:]

            num_last_layer = sum(list_lengths[-1:]) 
            # in FISH ResNet architecture, the last layer has bias=False
            # if last layer has both weight and bias, set -1 to -2 above

            if num_last_layer < num_top:
                top_idxs = np.hstack(
                    [sorted_idxs[-(num_top - num_last_layer):], 
                    np.arange(cum_lengths[-2], cum_lengths[-1])]
                )
                assert len(top_idxs) == num_top
            else:
                raise ValueError("too small top percentage")

            imp_wt_idxs = [[] for i in range(len(list_t))]
            for idx in top_idxs:
                prev_length = 0
                for idx_layer_num, length in enumerate(cum_lengths):
                    if idx < length and length > prev_length: 
                        try:
                            idx_tuple = np.nonzero(combined_arrays[idx] == list_t[idx_layer_num])
                            imp_wt_idxs[idx_layer_num].append(idx_tuple)
                        except Exception:
                            print("caught error: ", idx, idx_layer_num, length, imp_wt_idxs)
                            raise
                        break
                    prev_length = length
        elif method == "dispersed":
            grad_values = list(sq_grads_expect.values())
            imp_wt_idxs = [[] for i in range(len(grad_values))]
            for layer in range(len(grad_values)): #layer-by-layer
                num_imp = np.ceil(pct_top * np.prod(np.array(grad_values[layer]).shape))
                sorted_grads = np.argsort(grad_values[layer],axis=None)
                if layer == len(grad_values)-1:
                    top_grad_idxs = sorted_grads
                else:
                    top_grad_idxs = sorted_grads[-int(num_imp):]
                for idx in top_grad_idxs:
                    try:
                        imp_wt_idxs[layer].append(np.unravel_index(idx, grad_values[layer].shape))
                    except Exception:
                        print("caught error: ", layer, len(np.array(grad_values[layer]).flatten()), len(top_grad_idxs), top_grad_idxs)
                        raise
        elif method == "relative":
            grad_values = list(sq_grads_expect.values())
            imp_wt_idxs = [[] for i in range(len(grad_values))]
            for layer in range(len(grad_values)): #layer-by-layer
                layer_avg = np.average(grad_values[i])
                sorted_grads = np.argsort(grad_values[layer],axis=None)
                flat_layer = np.array(grad_values[layer]).flatten()
                top_grad_idxs = [i for i in sorted_grads if flat_layer[i] > layer_avg*1.25]
                for idx in top_grad_idxs:
                    try:
                        imp_wt_idxs[layer].append(np.unravel_index(idx, grad_values[layer].shape))
                    except Exception:
                        print("caught error: ", layer, len(np.array(grad_values[layer]).flatten()), len(top_grad_idxs), top_grad_idxs)
                        raise
        else:
            raise Exception("Invalid Fish Mask Selection Method.")
        return imp_wt_idxs


    



    def calculate_random_mask(self, mask_size=7014):
        num_params = sum(p.numel() for p in self.net.parameters())
        model_shape = []
        for i in self.net.parameters():
            model_shape.append(list(i.size()))

        flat_model_shape = []
        for i in self.net.parameters():
            flat_model_shape.append(np.prod(list(i.size())))
        cum_lengths = np.cumsum(flat_model_shape)
        possible_idxs = range(num_params)
        rand_wts = np.random.choice(possible_idxs, int(mask_size), replace=False)
        imp_wt_idxs = [[] for i in range(len(model_shape))]
        for i in rand_wts:
            prev_length = 0
            for idx_layer_num, length in enumerate(cum_lengths):
                if i < length and length > prev_length: 
                    try:
                        distance_into_layer = i-prev_length
                        layer_shape = model_shape[idx_layer_num]
                        idx_tuple = np.unravel_index(distance_into_layer, layer_shape)
                    except Exception:
                        print("caught error: ", i, idx_layer_num, prev_length, length, imp_wt_idxs)
                        raise
                    imp_wt_idxs[idx_layer_num].append(idx_tuple)
                    break
                prev_length = length
        return imp_wt_idxs

    """ return a tensor of size 50,000 x 10 x num imp idxs """
    def log_prob_grads_wrt(self, imp_idxs): 
        num_imp_per_layer = [len(t) for t in imp_idxs]
        log_prob_grads = np.zeros((len(self.Y), len(np.unique(self.Y)), sum(num_imp_per_layer)))

        '''
        makes a mask based on imp_idxs, initialize 62 arrays for the layers, these are the 62 masks
        '''
        masks_list = []
        for layer_num, layer_wt in enumerate(list(self.net.parameters())):
            mask = np.zeros_like(layer_wt.detach().cpu().numpy(), dtype=bool)
            for tup in imp_idxs[layer_num]:
                mask[tup] = True
            masks_list.append(mask)
        
        
        self.net.to(device)
        for param in self.net.parameters():
            param.requires_grad = True
        self.net.eval()
        parameters = tuple(self.net.parameters())
        test_loader = DataLoader(self.handler(self.X, self.Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        for test_batch, test_labels, idxs in test_loader:
            test_batch, test_labels = test_batch.cuda(), test_labels.cuda()
            
            outputs, e1 = self.net(test_batch)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1).to('cpu')
            log_probs = F.log_softmax(outputs, dim=1)
            N, C = log_probs.shape

            for n in range(N):
                for c in range(C):
                    grad_list = torch.autograd.grad(log_probs[n][c], parameters, retain_graph=True)
                    pos = 0
                    for i, grad in enumerate(grad_list):
                        grad = grad.detach().cpu().numpy()  # https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
                        selected_grads = grad[masks_list[i]] # before mask list, selected_grads = np.array([grad[t] for t in imp_idxs[i]]).reshape(-1)
                        log_prob_grads[idxs[n]][c][pos:(pos+len(imp_idxs[i]))] = selected_grads # log_prob_grads[image][class][slice corresp to ith layer] = selected_grads
                        pos += len(selected_grads)                       
                self.net.zero_grad()  
        return torch.tensor(log_prob_grads)          

    def query(self, n):
        self.fishIdentity == 0
        #imp_wt_idxs = self.calculate_fishmask(self.pct_top)
        if self.method != "random":
            imp_wt_idxs = self.calculate_fishmask(self.pct_top, self.method)
        else:
            imp_wt_idxs = self.rand_mask
        save_imp_weights(imp_wt_idxs, self.savefile)
        xt = self.log_prob_grads_wrt(imp_wt_idxs)
        torch.cuda.empty_cache()

        batchSize = 25
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
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
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
        
        xt = xt[idxs_unlabeled]
        
        chosen = select(xt, n, fisher, init, self.savefile, "FISH", lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb), chunkSize=self.chunkSize)
        save_queried_idx(idxs_unlabeled[chosen], self.savefile, self.alg)
        print('selected probs: ' +
                str(str(torch.mean(torch.max(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.min(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.std(phat[chosen,:], 1)).item())), flush=True)
        return idxs_unlabeled[chosen]