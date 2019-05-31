# Author: llw
import numpy as np
import torch as t
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, knn_graph

from utils.pooling import *


class GraphGlobal(t.nn.Module):
    def __init__(self, cfg):
        super(GraphGlobal, self).__init__()
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["out_channels"]
        self.mid_channels = cfg["mid_channels"]
        self.K = cfg["K"]
        self.chev_conv_front = ChebConv(in_channels=self.in_channels, out_channels=self.mid_channels, K=4)
        self.bn_front = t.nn.BatchNorm1d(self.mid_channels)
        self.dropout1 = t.nn.Dropout(p=cfg["drop_prob1"])
        self.chev_conv_back = ChebConv(in_channels=self.mid_channels, out_channels=self.mid_channels, K=3)
        self.bn_back = t.nn.BatchNorm1d(self.mid_channels)
        self.dropout2 = t.nn.Dropout(p=cfg["drop_prob2"])
        self.fc1 = t.nn.Linear(in_features=self.mid_channels * 4, out_features=600)
        self.fc2 = t.nn.Linear(in_features=600, out_features=self.out_channels)
        self.sftm = t.nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        front_out = self.chev_conv_front(x, edge_index)
        front_out = self.bn_front(front_out)
        front_out = F.relu(front_out)
        front_out = self.dropout1(front_out)
        back_out = self.chev_conv_back(front_out, edge_index)
        back_out = self.bn_back(back_out)
        back_out = F.relu(back_out)
        back_out = self.dropout1(back_out)
        front_feature = global_pooling(front_out)
        back_feature = global_pooling(back_out)
        feature = t.cat([front_feature, back_feature])
        feature = self.dropout2(feature)
        feature = self.fc1(feature)
        feature = F.relu(feature)
        feature = self.dropout2(feature)
        feature = self.fc2(feature)
        feature = self.sftm(feature)
        return feature.reshape([1, -1]).float()


if __name__ == '__main__':
    nn = GraphGlobal()
