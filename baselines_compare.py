import pickle
from torchvision import transforms
import argparse
from dataset import get_dataset, get_handler
import resnet


from saving import load_model
from experiments import exper


parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--savefile', help='name of file to save round accuracies to', type=str, default="compare")
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='CIFAR10')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=2000)
parser.add_argument('--fishIdentity', help='for ablation, setting fisher to be identity', type=int, default=0)
parser.add_argument('--fishInit', help='initialize selection with fisher on seen data', type=int, default=1)
parser.add_argument('--backwardSteps', help='openML dataset index, if any', type=int, default=1)
parser.add_argument('--lamb', help='lambda', type=float, default=1)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='resnet')
parser.add_argument('--pct_top', help='percentage of important weights to use for Fisher', type=float, default=0.01)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--chunkSize', help='for computation inside select function', type=int, default=200)

parser.add_argument('--compare', help='previous run to compare to', type=str, required=True, default='random_mask_exp_25K')

opts = parser.parse_args()
DATA_NAME = opts.data
SAVE_FILE = opts.savefile + '_for_' + opts.compare
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


_, __, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
dataset = pickle.load(open("Save/Queried_idxs/" + "dataset_" + opts.compare + ".p", "rb"))
X_tr, Y_tr = dataset['X_train'], dataset['Y_train']
handler = get_handler(DATA_NAME)

net = resnet.ResNet18()
idxs_lb = pickle.load(open("Save/Queried_idxs/" + "initLabeled_" + opts.compare + ".p", "rb"))
NUM_INIT_LB = sum(idxs_lb)
NUM_ROUND = int((len(X_tr) - NUM_INIT_LB)/ NUM_QUERY)
load_model(1, net, opts.compare, "BAIT") # load the checkpoint for rd 1 of BAIT

exper(opts.alg, X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)