import torch
import pickle
from torchvision import transforms
import numpy as np
import gc
#import openml
import os
import argparse
from dataset import get_dataset, get_handler
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
import time
from torchvision.models import resnet18,  ResNet18_Weights
import logging
from saving import save_accuracies, save_model, load_model
from DUQ_model import model_DUQ


from query_strategies import RandomSampling, BadgeSampling, FishEntSampling, \
                                 LeastConfidence, MarginSampling, \
                                EntropySampling, CoreSet, ActiveLearningByLearning, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                BaitSampling, fisher_mask_sampling, ResNetDUQEntropySampling

parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)  #1e-4
parser.add_argument('--paramScale', help='learning rate', type=float, default=1)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='CIFAR10')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=250) #70, #2000
parser.add_argument('--nStart', help='number of points to start', type=int, default=250) #15, #2000
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=5750) #925, 26000
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=128)
parser.add_argument('--rounds', help='number of embedding dims (mlp)', type=int, default=0)
parser.add_argument('--trunc', help='number of embedding dims (mlp)', type=int, default=-1)
parser.add_argument('--btype', help='acquisition algorithm', type=str, default='min')
parser.add_argument('--modes', help='openML dataset index, if any', type=int, default=1)
parser.add_argument('--aug', help='do augmentation (for cifar)', type=int, default=1)
parser.add_argument('--lamb', help='lambda', type=float, default=1)
parser.add_argument('--fishIdentity', help='for ablation, setting fisher to be identity', type=int, default=0)
parser.add_argument('--fishInit', help='initialize selection with fisher on seen data', type=int, default=1)
parser.add_argument('--backwardSteps', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--dummy', help='dummy input for indexing replicates', type=int, default=1)
parser.add_argument('--pct_top', help='percentage of important weights to use for Fisher', type=float, default=0.002)
parser.add_argument('--DEBUG', help='provide a size to utilize decreased dataset size for quick run', type=int, default=5750) #925 26000
parser.add_argument('--savefile', help='name of file to save round accuracies to', type=str, default="experiment0")
parser.add_argument('--chunkSize', help='for computation inside select function', type=int, default=200)
# parser.add_argument('--compare', help='previous run to compare to', type=str, required=True, default='random_mask_exp_25K')

opts = parser.parse_args()
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data
# logging.basicConfig(level=logging.DEBUG, filename=opts.savefile + '.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def decrease_dataset(X_tr, Y_tr):
    new_size = opts.DEBUG
    masks = [np.zeros(len(Y_tr), dtype = 'int') for i in range(10)]
    for i in range(len(Y_tr)):
        masks[Y_tr[i].item()][i] = 1

    # creating new_Xtr
    new_Xtr = []
    for i in masks:
        var = np.random.choice(np.arange(len(X_tr), dtype=int)[i == 1], int(new_size/10), replace=False)
        new_Xtr.append(X_tr[var])

    new_Xtr = np.array(new_Xtr).reshape(new_size, 32, 32, 3)
    # new_Xtr = np.array(new_Xtr).reshape(new_size, 3, 32, 32)

    # creating new_Ytr
    new_Ytr = np.zeros(new_size, dtype = 'int')
    slce = int(new_size/10)
    for i in range(10):
        new_Ytr[int(slce-new_size/10):slce] = i
        slce += int(new_size/10)
    return new_Xtr, torch.from_numpy(new_Ytr)
        
def exper(alg, X_tr, Y_tr, idxs_lb, net, handler, args,X_te, Y_te, DATA_NAME):  #, method="standard" at the end and model_DUQ after net
  

    time_begin_experiment = time.time()
    # set up the specified sampler
    if alg == 'BAIT': # bait sampling
        strategy = BaitSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'FISH': # fisher mask based sampling
        strategy = fisher_mask_sampling(X_tr, Y_tr, idxs_lb, net, handler, args) #method
    elif alg == 'rand':
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'coreset': # coreset sampling
        strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'entropy': # entropy-based sampling
        strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'kcent': # kcentergreedy-based sampling
        strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'margin': # kcentergreedy-based sampling
        strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'lcs': # kcentergreedy-based sampling
        strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif alg == 'FishEnt': # kcentergreedy-based sampling
        strategy = FishEntSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    # elif alg == 'DUQ':
        # strategy = ResNetDUQEntropySampling(X_tr, Y_tr, idxs_lb, net, model_DUQ, handler, args)
    else: 
        print('choose a valid acquisition function', flush=True)
        raise ValueError

    # print info
    if opts.did > 0: DATA_NAME='OML' + str(opts.did)
    print(DATA_NAME, flush=True)
    print(type(strategy).__name__, flush=True)

    if type(X_te) == torch.Tensor: X_te = X_te.numpy()

    # round 0 accuracy
    if alg == 'BAIT':
        strategy.train(verbose= False)
    else:
        strategy.clf = net.cuda()
    P = strategy.predict(X_te, Y_te)
    accur = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    save_accuracies(accur, opts.savefile, alg)
    print(str(opts.nStart) + '\ttesting accuracy {}'.format(accur), flush=True)

    for rd in range(1, NUM_ROUND+1):
        save_model(rd, net, opts.savefile, alg)
        print('Round {}'.format(rd), flush=True)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # query
        output = strategy.query(NUM_QUERY)
        q_idxs = output
        idxs_lb[q_idxs] = True

        # update
        update_time = time.time()
        strategy.update(idxs_lb)
        train_time = time.time()
        print('Update took:', train_time - update_time)
        strategy.train(verbose=False)

        # round accuracy
        predict_time = time.time()
        print('Train took:', predict_time - train_time)
        P = strategy.predict(X_te, Y_te)
        end_time = time.time()
        print('Predict took:', end_time - predict_time)
        accur = 1.0 * (Y_te == P).sum().item() / len(Y_te)
        save_accuracies(accur, opts.savefile, alg)
        print(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(accur), flush=True)
        if sum(~strategy.idxs_lb) < opts.nQuery: break
        if opts.rounds > 0 and rd == (opts.rounds - 1): break
        time_end_experiment = time.time()
        # logging.debug(f"{alg} experiment time:" + str(time_end_experiment - time_begin_experiment) + "seconds")

def main():
    # non-openml data defaults
    args_pool = {'MNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'FashionMNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.001, 'momentum': 0.5}},
                'SVHN':
                    {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.001, 'momentum': 0.3}},
                'CIFAR10':
                    {'n_epoch': 3, 'transform': transforms.Compose([ 
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                    ]),
                    'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 100, 'num_workers': 1}, # change back to 1000
                    'optimizer_args':{'lr': 0.001, 'momentum': 0.3},
                    'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}
                    }
    opts.nClasses = 10
    if opts.aug == 0: 
        args_pool['CIFAR10']['transform'] =  args_pool['CIFAR10']['transformTest'] # remove data augmentation
    args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
    args_pool['FashionMNIST']['transformTest'] = args_pool['FashionMNIST']['transform']
    args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']

    if opts.did == 0: args = args_pool[DATA_NAME]
    if not os.path.exists(opts.path):
        os.makedirs(opts.path)


    # load openml dataset if did is supplied
    if opts.did > 0:
        data = pickle.load(open('oml/data_' + str(opts.did) + '.pk', 'rb'))['data']
        X = np.asarray(data[0])
        y = np.asarray(data[1])
        y = LabelEncoder().fit(y).transform(y)
        opts.nClasses = int(max(y) + 1)
        nSamps, opts.dim = np.shape(X)
        testSplit = .1
        inds = np.random.permutation(nSamps)
        X = X[inds]
        y = y[inds]


        split =int((1. - testSplit) * nSamps)
        while True:
            inds = np.random.permutation(split)
            if len(inds) > 50000: inds = inds[:50000]
            X_tr = X[:split]
            X_tr = X_tr[inds]
            X_tr = torch.Tensor(X_tr)

            y_tr = y[:split]
            y_tr = y_tr[inds]
            Y_tr = torch.Tensor(y_tr).long()

            X_te = torch.Tensor(X[split:])
            Y_te = torch.Tensor(y[split:]).long()

            if len(np.unique(Y_tr)) == opts.nClasses: break

        #changed te args batch size from 1000 to 100
        args = {'transform':transforms.Compose([transforms.ToTensor()]),
                'n_epoch':10,
                'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                'loader_te_args':{'batch_size': 100, 'num_workers': 1},
                'optimizer_args':{'lr': 0.01, 'momentum': 0},
                'transformTest':transforms.Compose([transforms.ToTensor()])}
        handler = get_handler('other')

    # load non-openml dataset
    else:

        X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
        opts.dim = np.shape(X_tr)[1:]
        handler = get_handler(opts.data)
        # if False:
        if opts.DEBUG:
            X_tr, Y_tr = decrease_dataset(X_tr, Y_tr)
            if not os.path.exists("./Save/Queried_idxs/"):
                os.makedirs("./Save/Queried_idxs") 
            data_dict = {'X_train':X_tr, 'Y_train': Y_tr}
            with open("./Save/Queried_idxs/dataset_" + opts.savefile + '.p', "wb") as savefile:
                pickle.dump(data_dict, savefile)

        if opts.trunc != -1:
            inds = np.random.permutation(len(X_tr))[:opts.trunc]
            X_tr = X_tr[inds]
            Y_tr = Y_tr[inds]
            inds = torch.where(Y_tr < 10)[0]
            X_tr = X_tr[inds]
            Y_tr = Y_tr[inds]
            opts.nClasses = int(max(Y_tr) + 1)

    args['lr'] = opts.lr
    args['modelType'] = opts.model
    args['fishIdentity'] = opts.fishIdentity
    args['fishInit'] = opts.fishInit
    args['lamb'] = opts.lamb
    args['backwardSteps'] = opts.backwardSteps
    args['pct_top'] = opts.pct_top
    args['savefile'] = opts.savefile
    args['chunkSize'] = opts.chunkSize

    n_pool = len(Y_tr)
    n_test = len(Y_te)
    print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
    print('number of testing pool: {}'.format(n_test), flush=True)

    # generate initial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
    init_labeled = np.copy(idxs_lb)
    if opts.DEBUG:
        with open("./Save/Queried_idxs/initLabeled_" + opts.savefile + '.p', "wb") as savefile:
            pickle.dump(init_labeled, savefile)

    # linear model class
    class linMod(nn.Module):
        def __init__(self, dim=28):
            super(linMod, self).__init__()
            self.dim = dim
            self.lm = nn.Linear(dim, opts.nClasses)
        def forward(self, x):
            x = x.view(-1, self.dim)
            out = self.lm(x)
            return out, x
        def get_embedding_dim(self):
            return self.dim

    # mlp model class
    class mlpMod(nn.Module):
        def __init__(self, dim, embSize=128, useNonLin=True):
            super(mlpMod, self).__init__()
            self.embSize = embSize
            self.dim = int(np.prod(dim))
            self.lm1 = nn.Linear(self.dim, embSize)
            self.lm2 = nn.Linear(embSize, embSize)
            self.linear = nn.Linear(embSize, opts.nClasses, bias=False)
            self.useNonLin = useNonLin
        def forward(self, x):
            x = x.view(-1, self.dim)
            if self.useNonLin: emb = F.relu(self.lm1(x))
            else: emb = self.lm1(x)
            out = self.linear(emb)
            return out, emb
        def get_embedding_dim(self):
            return self.embSize

    # load specified network
    if opts.model == 'mlp':
        net = mlpMod(opts.dim, netembSize=opts.nEmb)
    elif opts.model == 'resnet':
        # net = resnet.ResNet18()
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 10)
    elif opts.model == 'vgg':
        net = vgg.VGG('VGG16')
    elif opts.model == 'lin':
        dim = np.prod(list(X_tr.shape[1:]))
        net = linMod(dim=dim)
    else: 
        print('choose a valid model - mlp, resnet, or vgg', flush=True)
        raise ValueError

    if opts.did > 0 and opts.model != 'mlp':
        print('openML datasets only work with mlp', flush=True)
        raise ValueError

    # if type(X_tr[0]) is not np.ndarray:
    #     X_tr = X_tr.numpy()
    
    # if type(X_tr[0]) is not torch.tensor:
    #     X_tr = torch.from_numpy(X_tr)

    # exper("entropy", X_tr, Y_tr, idxs_lb, net, model_DUQ, handler, args, X_te, Y_te, DATA_NAME)
    # # reset variables
    # idxs_lb = init_labeled
    # load_model(1, net, opts.savefile, "BAIT") # load the checkpoint for rd 1 of BAIT
    # ----------
    exper("DUQ", X_tr, Y_tr, idxs_lb, net, model_DUQ, handler, args, X_te, Y_te, DATA_NAME)
    #with open("./Save/Round_accuracies/Accuracy_for_" + opts.savefile + '.p', "r+b") as savefile:
        #acc_dict = pickle.load(savefile)
        #acc_dict['BAIT_time'] = bait_time
        #acc_dict['FISH_time'] = fish_time
        #pickle.dump(acc_dict, savefile)

if __name__=="__main__":
    main()
    
    
