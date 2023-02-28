import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import time
from .strategy import Strategy
import gc
import sys
import pickle

from .bait_sampling import select

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_mask(sq_grads_expect, pct_top=0.02):
    list_t = list(sq_grads_expect.values())
    combined_arrays = np.hstack([t.flatten() for t in sq_grads_expect.values()]) 
    list_lengths = [len(ten.flatten()) for ten in list_t]
    cum_lengths = np.cumsum(list_lengths)
    sorted_idxs = np.argsort(combined_arrays)
    num_top = int(pct_top * len(combined_arrays))
    top_idxs = sorted_idxs[-num_top:]

    imp_wt_idxs = [[] for i in range(len(list_t))]
    for idx in top_idxs:
        prev_length = 0
        for idx_layer_num, length in enumerate(cum_lengths):
            if idx < length and length > prev_length: 
                # print(len_idx)
                try:
                    # s_num[len_idx].append(np.where(combined_s[idx] == list_s[len_idx])[0][0])
                    idx_tuple = np.nonzero(combined_arrays[idx] == list_t[idx_layer_num])
                    '''pass only numpy or python objects to numpy functions'''
                    # s_num[len_idx].append([idx[0] for idx in idx_tuple])
                    imp_wt_idxs[idx_layer_num].append(idx_tuple)
                except IndexError:
                    print("caught error: ", idx, idx_layer_num, length, imp_wt_idxs)
                break
            prev_length = length
    return imp_wt_idxs


class fisher_mask_sampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(fisher_mask_sampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.fishIdentity = args['fishIdentity']
        self.fishInit = args['fishInit']
        self.lamb = args['lamb']
        self.backwardSteps = args['backwardSteps']
        self.pct_top = args['pct_top']

    def log_prob_grads_wrt(self, imp_idxs):
        '''
        return a tensor of size 50,000 x 10 x num imp idxs
        '''
        num_imp_per_layer = [len(t) for t in imp_idxs]
        log_prob_grads = np.zeros((len(self.Y), len(np.unique(self.Y)), sum(num_imp_per_layer))) 
        '''
        make a mask based on imp_idxs
        I have 62 layers of weights, and I can find their shapes
        initialize 62 different arrays using np.zeros(weight.shape), these are the 62 masks
        for i in range(len(imp_idxs)):
            for tup in imp_idxs[i]:
                mask_i[tup] = 1
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
        
        test_loader = DataLoader(self.handler(self.X, self.Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args']) # 'transformTest'
        
        idx = 0
        num_samples = 1
        for test_batch, test_labels, idxs in test_loader:
            """ if idx >= num_samples:
                break """
            test_batch, test_labels = test_batch.cuda(), test_labels.cuda()
            
            outputs, e1 = self.net(test_batch)
            _, preds = torch.max(outputs, 1)
            # print(torch.sum(preds == test_labels.data) / len(test_labels))
            # print('Batch: ', (idxs.numpy()[-1]+ 1)/len(test_batch) )

            probs = F.softmax(outputs, dim=1).to('cpu')
            log_probs = F.log_softmax(outputs, dim=1)
            N, C = log_probs.shape

            for n in range(N):
                if n % 10 == 0: 
                    TIME = False
                else: 
                    TIME = False
                for c in range(C):
                    start = time.time()
                    grad_list = torch.autograd.grad(log_probs[n][c], parameters, retain_graph=True) # ~0.007 secs
                    first = time.time()
                    
                    pos = 0
                    accum_grad = 0
                    accum_select = 0
                    accum_assign = 0
                    for i, grad in enumerate(grad_list):    # different layers # ~0.2 secs ~ 0.003 secs per iteration
                        start_i = time.time()
                        grad = grad.detach().cpu().numpy()  # https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
                        grad_time = time.time()
                        accum_grad += grad_time - start_i
                        if i % 10 == 0:
                            TIME_2 = False
                        else: 
                            TIME_2 = False
                        
                        
                        # selected_grads = np.array([grad[t] for t in imp_idxs[i]]).reshape(-1) 
                        selected_grads = grad[masks_list[i]]
                        select_time = time.time()
                        accum_select += select_time - grad_time
                        '''
                        trying to remove above for loop
                        use the 62 calculated masks as 
                        selected_grads = grads[mask_i]
                        '''
                        # selected_grads = grad[mask_i].reshape(-1)
                        
                        log_prob_grads[idxs[n]][c][pos:(pos+len(imp_idxs[i]))] = selected_grads
                        assign_time = time.time()
                        accum_assign += assign_time - select_time
                        pos += len(selected_grads)
                        
                        # if TIME_2: print('selection', first_2 - start_2, 'assignment', second_2 - first_2)
                        # selected_grads = np.array([grad[t] for t in imp_idxs[i]])
                        # log_prob_grads[image][class][slice corresp to ith layer] = selected_grads
                    second = time.time()
                    if TIME: 
                        print('grads', first - start, 'loop:', second - first)
                        print("Avg time for grad", accum_grad / (i+1), "select", accum_select / (i+1), "assign", accum_assign/(i+1))
                idx += 1
                self.net.zero_grad()
                """ if idx >= num_samples:
                    break """

          
        #return torch.tensor(log_prob_grads)          
        return log_prob_grads


    def query(self, n):
        self.fishIdentity == 0
        sq_grads_start = time.time()
        sq_grads_expect = self.calculate_gradients()
        imp_wt_start = time.time()
        #logging.info('calculate_gradients took ', imp_wt_start-sq_grads_start, ' seconds.')
        print('calculate_gradients took ', imp_wt_start-sq_grads_start, ' seconds.')
        imp_wt_idxs = calculate_mask(sq_grads_expect, pct_top=self.pct_top)
        xt_start = time.time()
        #logging.info('calculate_mask took ', xt_start-imp_wt_start, ' seconds.')
        print('calculate_mask took ', xt_start-imp_wt_start, ' seconds.')
        xt = self.log_prob_grads_wrt(imp_wt_idxs)
        xt_end = time.time()
        #logging.info('log_prob_grads_wrt took ', xt_end-xt_start, ' seconds.')
        print('log_prob_grads_wrt took ', xt_end-xt_start, ' seconds.')
        # with open('logprobgrads.pickle', 'wb') as f: 
        #     pickle.dump(xt, f)

        torch.cuda.empty_cache()
        # get fisher
        if self.fishIdentity == 0:
            print('getting fisher matrix ...', flush=True)
            time_long = time.time()
            batchSize = 3
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1]).cuda()
            rounds = int(np.ceil(len(self.X) / batchSize))
            for i in range(int(np.ceil(len(self.X) / batchSize))):
                '''
                adding individual fisher matrices to compute overall fisher matrix I(theta_^L_t)
                '''
                # print('Fisher for all', i)
                # xt_ = torch.tensor(xt[i * batchSize : (i + 1) * batchSize]).cuda()
                xt_ = torch.tensor(xt[i * batchSize : (i + 1) * batchSize]).cuda()
                # print(sys.getsizeof(xt_.flatten()))
                # print(xt_.size())
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0)#.detach().cpu()
                if(i%1000==0):
                    print(i/1000,'/',rounds/1000)
                fisher = fisher + op
                # xt_ = xt_.cpu()
                del xt_, op
                torch.cuda.empty_cache()
                gc.collect()
        else: fisher = torch.eye(xt.shape[-1])
        
        # get fisher only for samples that have been seen before
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        
        batchSize = 3
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        rounds = int(np.ceil(len(xt2) / batchSize))
        if self.fishInit == 1:
            sec_time = time.time()
            for i in range(int(np.ceil(len(xt2) / batchSize))):
                
                xt_ = torch.tensor(xt2[i * batchSize : (i + 1) * batchSize]).cuda()
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
                init = init + op
                xt_ = xt_.cpu()
                del xt_, op
                torch.cuda.empty_cache()
                gc.collect()

        sec_time_end = time.time()
        # print('Second for loop took:', sec_time_end-sec_time,'seconds')
        phat = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        time_long_end = time.time()
        print('Iterations took:',time_long_end - time_long,'seconds')
        print('all probs: ' + 
                str(str(torch.mean(torch.max(phat, 1)[0]).item())) + ' ' + 
                str(str(torch.mean(torch.min(phat, 1)[0]).item())) + ' ' + 
                str(str(torch.mean(torch.std(phat,1)).item())), flush=True)
        
        xt = xt[idxs_unlabeled]
        start_for_select = time.time()
        chosen = select(xt, n, fisher, init, lamb=self.lamb, backwardSteps=self.backwardSteps, nLabeled=np.sum(self.idxs_lb))
        end_for_select = time.time()
        print ('Select took', end_for_select - start_for_select, 'seconds')
        print('selected probs: ' +
                str(str(torch.mean(torch.max(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.min(phat[chosen, :], 1)[0]).item())) + ' ' +
                str(str(torch.mean(torch.std(phat[chosen,:], 1)).item())), flush=True)
        end = time.time()
        print('The rest of the query function took ', end-xt_end, ' seconds.')
        print('Query took ', (imp_wt_start-sq_grads_start) + (xt_start-imp_wt_start) + (xt_end-xt_start) + (end-xt_end))
        return idxs_unlabeled[chosen]
        

    def calculate_gradients(self):
        self.net.to(device)
        for param in self.net.parameters():
            param.requires_grad = True
        self.net.eval()
        parameters = tuple(self.net.parameters())
        sq_grads_expect = {i: np.zeros(p.shape) for i, p in enumerate(parameters)}

        """ preprocess = ResNet18_Weights.DEFAULT.transforms()
        processed_testset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=preprocess)
        test_loader = DataLoader(processed_testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) """
        test_loader = DataLoader(self.handler(self.X, self.Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args']) # 'transformTest'
        idx = 0
        num_samples = 1024 # used by FISH mask paper

        for test_batch, test_labels, idxs in test_loader:
            if idx >= num_samples:
                break
            
            test_batch, test_labels = test_batch.cuda(), test_labels.cuda()
            # print('idx: ', idx)
            #model.zero_grad() moved this line
            outputs, e1 = self.net(test_batch)
            _, preds = torch.max(outputs, 1)
            # print(torch.sum(preds == test_labels.data) / len(test_labels))
            # print('Batch: ', (idxs.numpy()[-1]+ 1)/len(test_batch) )
            
            probs = F.softmax(outputs, dim=1).to('cpu')
            log_probs = F.log_softmax(outputs, dim=1)
            N, C = log_probs.shape

            for n in range(N):
                # print('calc grads N: ', n)
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
            
        return sq_grads_expect