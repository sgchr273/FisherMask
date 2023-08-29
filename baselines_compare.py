import pickle
from torchvision import transforms
import argparse
from dataset import get_dataset, get_handler
import resnet
import os
import numpy as np
import torch
import torch.nn as nn
import time


from saving import load_model, save_queried_idx
from experiments import exper, opts
if __name__=="__main__":

    DATA_NAME = opts.data
    SAVE_FILE = opts.savefile
    NUM_QUERY = opts.nQuery

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
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'SVHN':
                    {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'CIFAR10':
                    {'n_epoch': 3, 'transform': transforms.Compose([ 
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                    ]),
                    'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 100, 'num_workers': 1}, # change back to 1000
                    'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                    'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}
                    }
    args = args_pool[DATA_NAME]
    args['lr'] = opts.lr
    args['modelType'] = opts.model
    args['fishIdentity'] = opts.fishIdentity
    args['fishInit'] = opts.fishInit
    args['lamb'] = opts.lamb
    args['backwardSteps'] = opts.backwardSteps
    args['pct_top'] = opts.pct_top
    args['chunkSize'] = opts.chunkSize

    args['savefile'] = SAVE_FILE


    # _, __, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
    # dataset = pickle.load(open("Save/Queried_idxs/" + "dataset_" + opts.compare + ".p", "rb"))
    # X_tr, Y_tr = dataset['X_train'], dataset['Y_train']
    # handler = get_handler(DATA_NAME)


def unbalanced_train_dataset(X_tr, Y_tr):

    masks = [np.zeros(len(Y_tr), dtype = 'int') for i in range(10)]
    for i in range(len(Y_tr)):
        masks[Y_tr[i].item()][i] = 1

    var = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[0] == 1], 250, replace=False)


    var2 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[1] == 1], 5000, replace=False)

    var3 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[2] == 1], 250, replace=False)

    var4 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[3] == 1], 250, replace=False)
    new_Xtr = np.concatenate(( X_tr[var], X_tr[var2], X_tr[var3], X_tr[var4]))
  
    new_Ytr = np.zeros(len(new_Xtr), dtype = 'int')
    new_Ytr[0:250] = 0
    new_Ytr[250:5250] = 1
    new_Ytr[5250:5500] = 2
    new_Ytr[5500:5750] = 3

    return new_Xtr, torch.from_numpy(new_Ytr)

def unbalanced_test_dataset(X_tr, Y_tr):

    masks = [np.zeros(len(Y_tr), dtype = 'int') for i in range(10)]
    for i in range(len(Y_tr)):
        masks[Y_tr[i].item()][i] = 1

    var = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[0] == 1], 1000, replace=False)
    var2 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[1] == 1], 1000, replace=False)
    var3 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[2] == 1], 1000, replace=False)
    var4 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[3] == 1], 1000, replace=False)
    
    new_Xtr = np.concatenate((X_tr[var], X_tr[var2], X_tr[var3], X_tr[var4]))
    new_Ytr = np.zeros(len(new_Xtr), dtype = 'int')
    new_Ytr[0:1000] = 0
    new_Ytr[1000: 2000] = 1
    
    new_Ytr[2000: 3000] = 2
    new_Ytr[3000 : 4000] = 3
    return new_Xtr, torch.from_numpy(new_Ytr)

if __name__=="__main__":

    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
    opts.dim = np.shape(X_tr)[1:]
    handler = get_handler(opts.data)

    X_tr, Y_tr = unbalanced_train_dataset(X_tr, Y_tr)
    if not os.path.exists("./Save/Queried_idxs/"):
        os.makedirs("./Save/Queried_idxs") 
    data_dict = {'X_train':X_tr, 'Y_train': Y_tr}
    with open("./Save/Queried_idxs/dataset_" + opts.savefile + '.p', "wb") as savefile:
        pickle.dump(data_dict, savefile)

    X_te, Y_te = unbalanced_test_dataset(X_te, Y_te)
    n_pool = len(Y_tr)
    NUM_INIT_LB = opts.nStart

    net = resnet.ResNet18(num_classes=4)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        net = resnet.ResNet18(num_classes=4)
        load_model(1, net, opts.savefile, 'entropy')
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
        init_labeled = np.copy(idxs_lb)
        with open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + '.p', "wb") as savefile:
            pickle.dump(init_labeled, savefile)
        exper('entropy', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)

    net = resnet.ResNet18(num_classes=4)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        load_model(1, net, opts.savefile, 'entropy')
        idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
        exper('rand', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)

    net = resnet.ResNet18(num_classes=4)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        load_model(1, net, opts.savefile, 'entropy')
        idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
        exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="standard")

    net = resnet.ResNet18(num_classes=4)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        load_model(1, net, opts.savefile, 'entropy')
        idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
        exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="dispersed")

    net = resnet.ResNet18(num_classes=4)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        load_model(1, net, opts.savefile, 'entropy')
        idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
        exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="relative")
    
