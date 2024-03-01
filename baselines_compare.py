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
from experiments import exper, opts, decrease_dataset
# if __name__=="__main__":

DATA_NAME = opts.data
SAVE_FILE = opts.savefile
NUM_QUERY = opts.nQuery

# non-openml data defaults
args_pool = {'MNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                'optimizer_args':{'lr': 0.001, 'momentum': 0.5}},
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
                'loader_tr_args':{'batch_size': 500, 'num_workers': 1},  #large batchsize does not necessarily reduce the time of computation. It is kind of trade off that I should know.
                'loader_te_args':{'batch_size': 200, 'num_workers': 1}, # change back to 1000
                'optimizer_args':{'lr': 0.001, 'momentum': 0.3},
                'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}
                }
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['FashionMNIST']['transformTest'] = args_pool['FashionMNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']

# DATA_NAME = 'MNIST'
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

# print(args)

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

    var = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[0] == 1], 1000) #I changed these numbers to 500 instead of 1000 for MNIST because it was throiwng thta error of ValueError: Cannot take a larger sample than population when 'replace=False' 
    var2 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[1] == 1], 1000) # last argument for all of these should be replace=False
    var3 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[2] == 1], 1000)
    var4 = np.random.choice(np.arange(len(X_tr), dtype=int)[masks[3] == 1], 1000)
    
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
    # X_tr, Y_tr = decrease_dataset(X_tr, Y_tr)

    if not os.path.exists("./Save/Queried_idxs/"):
        os.makedirs("./Save/Queried_idxs") 
    data_dict = {'X_train':X_tr, 'Y_train': Y_tr}
    with open("./Save/Queried_idxs/dataset_" + opts.savefile + '.p', "wb") as savefile:
        pickle.dump(data_dict, savefile)

    X_te, Y_te = unbalanced_test_dataset(X_te, Y_te)
    n_pool = len(Y_tr)
    NUM_INIT_LB = opts.nStart

    net = resnet.ResNet18(num_classes=4)
    for i in range(1):
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
        exper('FishEnt', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 


    net = resnet.ResNet18(num_classes=4)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        load_model(1, net, opts.savefile, 'entropy')
        idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
        exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 

    # net = resnet.ResNet18(num_classes=4)
    # for i in range(5):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('rand', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 

    # # idxs_lb = np.zeros(n_pool, dtype=bool)
    # # idxs_tmp = np.arange(n_pool)
    # # np.random.shuffle(idxs_tmp)
    # # idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
    # net = resnet.ResNet18(num_classes=4)
    # for i in range(5):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     opts.savefile = SAVE_FILE + str(i) 
    #     exper('BAIT', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)  #, method="standard"
    # net = resnet.ResNet18(num_classes=4) 
    # for i in range(5):
    #     opts.savefile = SAVE_FILE+  str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile  +".p", "rb"))   
    #     exper('kcent', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)  #, method="standard"
    # net = resnet.ResNet18(num_classes=4)
    # for i in range(5):  
    #     opts.savefile = SAVE_FILE+  str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile  +".p", "rb"))   
    #     exper('lcs', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)  #, method="standard"
    # net = resnet.ResNet18(num_classes=4)
    # for i in range(5):
    #     # net = resnet.ResNet18(num_classes=4) 
    #     opts.savefile = SAVE_FILE+  str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile  +".p", "rb"))   
    #     exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)  #, method="standard"


    # net = resnet.ResNet18(num_classes=4)
    # for i in range(5):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     opts.savefile = SAVE_FILE + str(i) + "_dispersed"
    #     exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="dispersed") 

    # net = resnet.ResNet18(num_classes=4)
    # for i in range(5):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     opts.savefile = SAVE_FILE + str(i) + "_relative"
    #     exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="relative")
    

#lr didn't matter for MNIST. I only  got the error in unlabeled_test_dataset function because I was sampling more points than were available so I removed replace=False and then it worked pretty fine.
#lr only mattered for CIFAR10 where Fish_mask gave best perfromance with lr = 0.001