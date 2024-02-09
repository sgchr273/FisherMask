import torch
import torch.nn as nn
import torch.optim as optim
from .strategy import Strategy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# import torchvision.models as models, DenseNet121_Weights
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class ResNetDUQEntropySampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, resnet_duq_model, handler, args):
        super(ResNetDUQEntropySampling, self).__init__(X, Y, idxs_lb, net, resnet_duq_model, handler, args)
        # self.X = X
        # self.Y = Y
        # self.idxs_lb = idxs_lb
        # self.resnet_duq_model = resnet_duq_model
        # self.args = args
        # self.n_pool = len(X)

    def predict_uncertainty(self, inputs):
        with torch.no_grad():
            self.resnet_duq_model.eval()
            inputs = torch.from_numpy(inputs).permute(0,3,2,1)
            inputs = inputs/255
            uncertainty_scores = self.resnet_duq_model(inputs)
        return uncertainty_scores
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        idxs_unlab_torch = torch.tensor(idxs_unlabeled)

        # Assuming your data needs some transformation before passing it to ResNet_DUQ
        # transformed_inputs = torch.tensor([transform(Image.fromarray(img)) for img in self.X[idxs_unlabeled]])

        uncertainty_scores = self.predict_uncertainty(self.X[idxs_unlab_torch])
        # U = uncertainty_scores.sum(dim=0).mean(dim=1)  # Modify this based on your ResNet_DUQ model
        U = uncertainty_scores.sum(1)
        # _, uncertain_indices = U.sort(descending=True)
        # result = idxs_unlabeled[uncertain_indices[:n]]
        uncertainty_chosen = idxs_unlabeled[U.sort()[1][:n]]

        # return chosen


### code to combine DUQ and kmeans

    #     embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

    #     cluster_learner = KMeans(n_clusters=n)
    #     cluster_learner.fit(embedding)

    #     cluster_idxs = cluster_learner.predict(embedding)
    #     centers = cluster_learner.cluster_centers_[cluster_idxs]
    #     dis = (embedding - centers) ** 2
    #     dis = dis.sum(axis=1)
    #     q_idxs = np.array(
    #         [np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])

    #     clustering_chosen = idxs_unlabeled[q_idxs]
    #     uncertainty_half = uncertainty_chosen[:len(uncertainty_chosen)//2]

    #     # Select the first half of clustering_chosen
    #     clustering_half = clustering_chosen[:len(clustering_chosen)//2]

    #     # Combine both halves
    #     all_chosen = np.concatenate((uncertainty_half, clustering_half), axis=None)

    #     return all_chosen



    #####Code for querying indices based on cosine similarity
    
    # def query(self, n):
    #     # Extract feature vectors for labeled data
    #     labeled_feature_vectors = self.extract_feature_vectors(self.idxs_lb)

    #     # Extract feature vectors for unlabeled data
    #     unlabeled_feature_vectors = self.extract_feature_vectors(~self.idxs_lb)

    #     # Compute cosine similarity
    #     similarity_matrix = cosine_similarity(unlabeled_feature_vectors, labeled_feature_vectors)

    #     # Select data points with highest dissimilarity
    #     chosen_indices = np.argsort(similarity_matrix.mean(axis=1))[-n:]

    #     return chosen_indices

    # def extract_feature_vectors(self, mask):
    #     densenet_model = models.densenet121(weights=DenseNet121_Weights)
    #     densenet_model = densenet_model.features.cuda()
    #     densenet_model.eval()
    #     densenet_model = nn.DataParallel(densenet_model)
    #     feature_vectors = []
    #     for i in range(len(self.X)):
    #         if mask[i]:
    #             image_tensor = torch.tensor(self.X[i]).unsqueeze(0)
    #             image_tensor = image_tensor.permute(0,3,2,1).float().cuda()
    #             with torch.no_grad():
    #                 activations = densenet_model(image_tensor)
    #                 feature_vector = activations.view(activations.size(0), -1).cpu().numpy()
    #                 feature_vectors.append(feature_vector)

    #     return np.concatenate(feature_vectors)