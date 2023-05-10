import torch

from torch import nn
import torch.nn.functional as F

from backbone.resnets import resnet56

class Model(torch.nn.Module):

    def __init__(self, backbone=resnet56(), num_clusters=100, dim=64, random=False):
        super(Model, self).__init__()
        self.backbone = backbone
        self.backbone.fc = torch.nn.Identity()
        self.nv = NetVLAD(num_clusters=num_clusters, dim=dim, random=random)

    def forward(self, x):
        emb = self.backbone(x)                      # get residual features
        emb = emb.unsqueeze(-1).unsqueeze(-1)       # (NxD) -> (NxDx1x1)
        nv_enc = self.nv(emb)                       # encode features
        return F.normalize(nv_enc)                  # final normalization

class NetVLAD(nn.Module):
    """Net(R)VLAD layer implementation"""

    def __init__(self, num_clusters=100, dim=64, alpha=100.0, random=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            random : bool
                enables NetRVLAD, removes alpha-init and normalization

        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.random = random
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)

        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        if not self.random:
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if not self.random:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # x = self.pool(x)
        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        
        residual *= soft_assign.unsqueeze(2)

        vlad = residual.sum(dim=-1)

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        
        vlad = vlad.view(x.size(0), -1)  # flatten

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad