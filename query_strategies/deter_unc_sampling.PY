import torch
import torch.nn as nn
import torch.optim as optim
from .strategy import Strategy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class ResNetDUQEntropySampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, resnet_duq_model, handler, args):
        super(ResNetDUQEntropySampling, self).__init__(X, Y, idxs_lb, net, resnet_duq_model, handler, args)
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.resnet_duq_model = resnet_duq_model
        self.args = args
        self.n_pool = len(X)

    def predict_uncertainty(self, inputs):
        with torch.no_grad():
            self.resnet_duq_model.eval()
            uncertainty_scores = self.resnet_duq_model.rbf(inputs)
        return uncertainty_scores

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # Assuming your data needs some transformation before passing it to ResNet_DUQ
        # transformed_inputs = torch.tensor([transform(Image.fromarray(img)) for img in self.X[idxs_unlabeled]])

        uncertainty_scores = self.predict_uncertainty(self.X[idxs_unlabeled])
        U = uncertainty_scores.sum(dim=0).mean(dim=1)  # Modify this based on your ResNet_DUQ model

        _, uncertain_indices = U.sort(descending=True)
        result = idxs_unlabeled[uncertain_indices[:n]]

        return result
