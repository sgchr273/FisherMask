from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
from torchvision.models import resnet18, ResNet18_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Using pretrained weights:
model = resnet18(weights=ResNet18_Weights.DEFAULT)
preprocess = ResNet18_Weights.DEFAULT.transforms()

# Partitioning and managing datatset
processed_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
class_names = processed_trainset.classes
processed_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
dataloaders = {'train' : [], 'val' : []}
dataloaders['train'] = torch.utils.data.DataLoader(processed_trainset, batch_size=100,
                                                    shuffle=True, num_workers=4, pin_memory=True)
dataloaders['val'] = torch.utils.data.DataLoader(processed_testset, batch_size=100,
                                                    shuffle=False, num_workers=4, pin_memory=True)
dataset_sizes = {'train' : len(processed_trainset), 'val' : len(processed_testset)}


# Setting last parameters to True and rest of them to False
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 10)
model = model.cpu()

# Setting critenion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# training model and calculating accuracy for last-layer parameters
model_conv, hist = train_model(model, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=50)

# Setting alll parameters back to true
for param in model.parameters():
    param.requires_grad = True


# Calculating the squared gradients
model.eval()
parameters = tuple(model.parameters())
sq_grads_expect = {i: np.zeros(p.shape) for i, p in enumerate(parameters)} # 

for test_batch, test_labels in dataloaders['val']:
    model.zero_grad()
    outputs = model(test_batch)
    _, preds = torch.max(outputs, 1)
    print(torch.sum(preds == test_labels.data) / len(test_labels))
    
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


# Manipulation the calcualted gradients
list_t = list(sq_grads_expect.values())
combined_arrays = np.hstack([t.flatten() for t in sq_grads_expect.values()]) 
list_lengths = [len(ten.flatten()) for ten in list_t]
cum_lengths = np.cumsum(list_lengths)
sorted_idxs = np.argsort(combined_arrays)
num_top = int(0.02 * len(combined_arrays))
top_idxs = sorted_idxs[-num_top:]

# calculating important weights
t_num = [[] for i in range(len(list_t))]
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
                t_num[idx_layer_num].append(idx_tuple)
            except IndexError:
                print("caught error: ", idx, idx_layer_num, length, t_num)
            break
        prev_length = length


# Plotting the important 2% weights
import matplotlib.pyplot as plt
# plt.plot([len(t) for t in t_num]) # gets biased by number of weights in a layer
plt.plot([len(t) / len(g.flatten()) for t, g in zip(t_num, list_t)], 'o')
plt.xlabel('layer number')
plt.ylabel('fraction of weights considered important in a layer')
plt.title('threshold for important weights is 0.02')
plt.show()