
import torch
import torch.nn as nn
from torchvision.models import resnet18
class ResNet_DUQ(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self.feature_extractor(x)
        y_pred = self.rbf(z)

        return y_pred
    
model_output_size = 512
epochs = 100
milestones = [25, 50, 75]
feature_extractor = resnet18()

# Adapted resnet from:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
feature_extractor.conv1 = torch.nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
feature_extractor.maxpool = torch.nn.Identity()
feature_extractor.fc = torch.nn.Identity()

centroid_size = model_output_size
num_classes = 10
length_scale = 0.1
gamma =0.999


model_DUQ = ResNet_DUQ(
feature_extractor,
num_classes,
centroid_size,
model_output_size,
length_scale,
gamma)