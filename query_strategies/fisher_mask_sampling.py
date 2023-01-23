import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import time
from .strategy import Strategy

from .bait_sampling import select

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_mask(sq_grads_expect):
    list_t = list(sq_grads_expect.values())
    combined_arrays = np.hstack([t.flatten() for t in sq_grads_expect.values()]) 
    list_lengths = [len(ten.flatten()) for ten in list_t]
    cum_lengths = np.cumsum(list_lengths)
    sorted_idxs = np.argsort(combined_arrays)
    num_top = int(0.02 * len(combined_arrays))
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

    def log_prob_grads_wrt(self, imp_idxs):
        '''
        return a tensor of size 60,000 x 10 x num imp idxs
        '''
        num_imp_per_layer = [len(t) for t in imp_idxs]
        log_prob_grads = np.zeros((len(self.Y), len(np.unique(self.Y)), sum(num_imp_per_layer))) 
        
        self.net.to(device)
        for param in self.net.parameters():
            param.requires_grad = True
        self.net.eval()
        parameters = tuple(self.net.parameters())
        
        test_loader = DataLoader(self.handler(self.X, self.Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args']) # 'transformTest'
        
        for test_batch, test_labels, idxs in test_loader:
            test_batch, test_labels = test_batch.cuda(), test_labels.cuda()
            
            outputs, e1 = self.net(test_batch)
            _, preds = torch.max(outputs, 1)
            print(torch.sum(preds == test_labels.data) / len(test_labels))
            print(idxs)

            probs = F.softmax(outputs, dim=1).to('cpu')
            log_probs = F.log_softmax(outputs, dim=1)
            N, C = log_probs.shape

            for n in range(N):
                print('N: ', n)
                for c in range(C):
                    start = time.time()
                    grad_list = torch.autograd.grad(log_probs[n][c], parameters, retain_graph=True)
                    first = time.time()
                    pos = 0
                    for i, grad in enumerate(grad_list):    # different layers
                        grad = grad.cpu()
                        selected_grads = np.array([grad[t].item() for t in imp_idxs[i]])
                        log_prob_grads[idxs[n]][c][pos:(pos+len(imp_idxs[i]))] = selected_grads
                        pos += len(selected_grads)
                        # selected_grads = np.array([grad[t] for t in imp_idxs[i]])
                        # log_prob_grads[image][class][slice corresp to ith layer] = selected_grads
                    second = time.time()
                    print('grads', first - start, 'loop:', second - first)
                    self.net.zero_grad()
                    
                    
        
        return torch.Tensor(log_prob_grads)


    def query(self, n):
        sq_grads_expect = self.calculate_gradients()
        imp_wt_idxs = calculate_mask(sq_grads_expect)
        xt = self.log_prob_grads_wrt(imp_wt_idxs)

        # get fisher
        if self.fishIdentity == 0:
            print('getting fisher matrix ...', flush=True)
            batchSize = 1000
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
            rounds = int(np.ceil(len(self.X) / batchSize))
            for i in range(int(np.ceil(len(self.X) / batchSize))):
                '''
                adding individual fisher matrices to compute overall fisher matrix I(theta_^L_t)
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
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        
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
        num_samples = 1 #1024 # used by FISH mask paper

        for test_batch, test_labels, idxs in test_loader:
            if idx >= num_samples:
                break
            test_batch, test_labels = test_batch.cuda(), test_labels.cuda()
            # print('idx: ', idx)
            #model.zero_grad() moved this line
            outputs, e1 = self.net(test_batch)
            _, preds = torch.max(outputs, 1)
            print(torch.sum(preds == test_labels.data) / len(test_labels))
            
            probs = F.softmax(outputs, dim=1).to('cpu')
            log_probs = F.log_softmax(outputs, dim=1)
            N, C = log_probs.shape

            for n in range(N):
                print('calc grads N: ', n)
                for c in range(C):
                    grad_list = torch.autograd.grad(log_probs[n][c], parameters, retain_graph=True)
                    for i, grad in enumerate(grad_list):    # different layers
                        gsq = torch.square(grad).to('cpu') * probs[n][c] / N
                        sq_grads_expect[i] += gsq.detach().numpy() # sq_grads_expect[i] + gsq
                        del gsq
                    self.net.zero_grad()
            idx += 1
        return sq_grads_expect