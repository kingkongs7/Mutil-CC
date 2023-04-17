import torch.nn as nn
import torch
from torch.nn.functional import normalize 
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models
from typing import Optional, TypeVar

__all__ = [
    'Network_Mul',
]

class Network_Mul(nn.Module):
    def __init__(self, feature_dim, class_num, num_head):
        super(Network_Mul, self).__init__()

        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.num_head = num_head

        # resnet
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # mutil head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead())

        self.instance_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim)
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward_feature2head(self, x):
        x = self.features(x)  # torch.Size([64, 512, 7, 7])
        # man
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))  # 4 * torch.Size([64, 512])

        # AFN
        heads = torch.stack(heads).permute([1, 0, 2])  # torch.Size([64, 4, 512])
        if heads.size(1)>1:
            heads = F.log_softmax(heads, dim=1)  # torch.Size([64, 4, 512])
        out = F.normalize(heads.sum(dim=1), dim=1)
        return out, heads

    def forward(self, x1, x2, x3:Optional[TypeVar] = None):
        h1, heads = self.forward_feature2head(x1)
        h2, _ = self.forward_feature2head(x2)
        
        # z1 = F.normalize(self.instance_projector(h1), dim=1)
        # z2 = F.normalize(self.instance_projector(h2), dim=1)
        z1 = h1
        z2 = h2

        c1 = self.cluster_projector(h1)
        c2 = self.cluster_projector(h2)

        if x3 != None:
            h3, _ = self.forward_feature2head(x3)
            # z3 = F.normalize(self.instance_projector(h3), dim=1)
            z3 = h3
            c3 = self.cluster_projector(h3)
            return z1, z2, z3, c1, c2, c3, heads
        else:
            return z1, z2, c1, c2, heads

    def forward_cluster(self, x):
        h, _ = self.forward_feature2head(x)
        c = self.cluster_projector(h)
        return c


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca  # 


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),  # 256个 1*1*512 卷积
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        return out 


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        return out
