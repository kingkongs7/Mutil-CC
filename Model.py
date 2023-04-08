import torch.nn as nn
import torch
from torch.nn.functional import normalize 
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models
from typing import Optional, TypeVar

__all__ = [
    'Network_CC',
]

class Network_CC(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(Network_CC, self).__init__()

        self.feature_dim = feature_dim
        self.cluster_num = class_num

        # resnet
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )

        # instance
        self.instance_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim)
        )

        # cluster 
        self.cluster_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x1, x2, x3:Optional[TypeVar] = None):
        h1 = self.features(x1)
        h2 = self.features(x2)
        
        z1 = F.normalize(self.instance_projector(h1), dim=1)
        z2 = F.normalize(self.instance_projector(h2), dim=1)

        c1 = self.cluster_projector(h1)
        c2 = self.cluster_projector(h2)
        if x3 != None:
            h3 = self.features(x3)
            z3 = F.normalize(self.instance_projector(h3), dim=1)
            c3 = self.cluster_projector(h3)
            return z1, z2, z3, c1, c2, c3
        else:
            return z1, z2, c1, c2
    
    def forward_cluster(self, x):
        h = self.features(x)
        c = self.cluster_projector(h)
        return c