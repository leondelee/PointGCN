# Author: llw
import torch.nn.functional as F
# from torch_geometric.nn import ChebConv

from utils.pooling import *
from model.cheb_conv import *


class GraphGlobal(t.nn.Module):
    def __init__(self, cfg):
        super(GraphGlobal, self).__init__()
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["out_channels"]
        self.mid_channels = cfg["mid_channels"]
        self.cheb_conv_front = ChebConv(self.in_channels, self.mid_channels, K=cfg["first_cheb_order"])
        self.cheb_conv_back = ChebConv(self.mid_channels, self.mid_channels, K=cfg["second_cheb_order"])
        self.fc1 = t.nn.Linear(in_features=self.mid_channels * 4, out_features=600)
        self.fc2 = t.nn.Linear(in_features=600, out_features=self.out_channels)
        self.dropout1 = t.nn.Dropout(p=cfg["drop_prob1"])
        self.dropout2 = t.nn.Dropout(p=cfg["drop_prob2"])

    def forward(self, x, lap):
        front_out = self.cheb_conv_front(x, lap)
        front_out = F.relu(front_out)
        front_out = self.dropout1(front_out)
        back_out = self.cheb_conv_back(front_out, lap)
        back_out = F.relu(back_out)
        back_out = self.dropout1(back_out)
        front_feature = global_pooling(front_out)
        back_feature = global_pooling(back_out)
        # front_feature = front_out
        # back_feature = back_out
        feature = t.cat([front_feature, back_feature], dim=-1)
        feature = self.dropout2(feature)
        feature = self.fc1(feature)
        feature = F.relu(feature)
        feature = self.dropout2(feature)
        feature = self.fc2(feature)
        return feature


if __name__ == '__main__':
    import numpy as np
    from sklearn.cluster import KMeans
    kms = KMeans(n_clusters=10, n_jobs=10)
    a = np.random.randn(125526, 3)
    kms.fit(a)
    lbs = kms.labels_
    mask0 = lbs == 0
    print(np.sum(mask0))
    cls0 = a[mask0]
    print(cls0.shape)




