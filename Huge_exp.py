

import pandas as pd
import numpy as np
from experiments import exper
from dataset import get_dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.models import resnet18,  ResNet18_Weights
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
import resnet
import vgg
from dataset import get_dataset, get_handler

def load_data(dat):
    if dat.isnumeric():
        data = pickle.load(open('oml/data_' + dat + '.pk', 'rb'))['data']
        X = np.asarray(data[0])
        y = np.asarray(data[1])
        y = LabelEncoder().fit(y).transform(y)
        nClasses = int(max(y) + 1)
        nSamps, dim = np.shape(X)
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
            if len(np.unique(Y_tr)) == nClasses: break
        args = {'transform':transforms.Compose([transforms.ToTensor()]),
                'n_epoch':10,
                'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                'loader_te_args':{'batch_size': 100, 'num_workers': 1},
                'optimizer_args':{'lr': 0.01, 'momentum': 0},
                'transformTest':transforms.Compose([transforms.ToTensor()])}
        handler = get_handler('other')
    else:
        nClasses = 10
        X_tr, Y_tr, X_te, Y_te = get_dataset(dat, path='data')
        dim = np.shape(X_tr)[1:]
        handler = get_handler(dat)
        args = {'MNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
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
        args[dat]['transformTest'] = args['MNIST']['transform']
        args[dat]['lr'] = 0.001
        # args['modelType'] = opts.model
        args[dat]['fishIdentity'] = 0
        args[dat]['fishInit'] = 1
        args[dat]['lamb'] = 1
        args[dat]['backwardSteps'] = 0
        args[dat]['pct_top'] = 0.002
        args[dat]['savefile'] = 'dummy'
        args[dat]['chunkSize'] = 200
    return X_tr, Y_tr, X_te, Y_te, dim, nClasses, handler, args[dat]


# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, nClasses, embSize=128, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, embSize)
        self.linear = nn.Linear(embSize, nClasses, bias=False)
        self.useNonLin = useNonLin
    def forward(self, x):
        x = x.view(-1, self.dim)
        if self.useNonLin: emb = F.relu(self.lm1(x))
        else: emb = self.lm1(x)
        out = self.linear(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize



def load_model(model, dim, nClasses):
    if model == 'mlp':
        net = mlpMod(dim, nClasses)
    elif model == 'rn':
        net = resnet.ResNet18()
        # net = resnet18(weights=ResNet18_Weights.DEFAULT)
        # num_ftrs = net.fc.in_features
        # net.fc = nn.Linear(num_ftrs, 10)
    elif model == 'vgg':
        net = vgg.VGG('VGG16')
    return net


datasets = ['MNIST', 'SVHN', 'CIFAR10', '6', '155', '156', '184']
models = ['mlp', 'rn', 'vgg']
DApair = [('MNIST','rn'), ('CIFAR10','rn'), ('155','mlp'), ('6','mlp'), ('156','mlp'),
          ('SVHN','rn'), ('SVHN', 'vgg'), ('CIFAR10','vgg'), ('MNIST', 'vgg'), ('184','mlp')]

algs = ['entropy', 'rand', 'kcent', 'margin', 'BAIT', 'FISH']
TrainingAug = 1
nQueries = [100, 1000, 5000]
# nQueries = [50, 100, 150]
reps = [1, 2, 3, 4, 5]



# Generate random values for each combination
for rep in reps:
    for pair in DApair:
        X_tr, Y_tr, X_te, Y_te, dim, nClasses, handler, args = load_data(pair[0])
        n_pool = len(Y_tr)
        NUM_INIT_LB = 50
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        for alg in algs:
            for nQuery in nQueries:
                net = load_model(pair[1], dim, nClasses)
                idxs_lb = np.zeros(n_pool, dtype=bool)
                idxs_lb[idxs_tmp[:int(nQuery/2)]] = True
                exper(alg, X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, pair[0], rep, nQuery)
                    
# # Create a DataFrame
# df = pd.DataFrame(data, columns=['Data', 'Model', 'Alg', 'nQuery', 'TrainAug', 'Rep', 'Samples', 'Accuracy', 'Time'])

# # Display the DataFrame
# print(df)