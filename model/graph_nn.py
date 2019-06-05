# Author: llw
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, GCNConv

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
        self.fc1.weight = t.nn.Parameter(
            t.normal(mean=t.zeros(600, self.mid_channels * 4).float(), std=cfg["init_linear_std"] * t.ones(600, self.mid_channels * 4)).float()
        )
        self.fc1.bias = t.nn.Parameter(
            t.normal(mean=t.zeros(600).float(), std=cfg["init_linear_std"] * t.ones(600).float())
        )
        self.fc2.weight = t.nn.Parameter(
            t.normal(mean=t.zeros(self.out_channels, 600).float(), std=cfg["init_linear_std"] * t.ones(self.out_channels, 600).float())
        )
        self.fc2.bias = t.nn.Parameter(
            t.normal(mean=t.zeros(self.out_channels).float(), std=cfg["init_linear_std"] * t.ones(self.out_channels).float())
        )
        # self.chev_conv_front = ChebConv(in_channels=self.in_channels, out_channels=self.mid_channels, K=cfg["first_cheb_order"])
        # self.chev_conv_front.weight = t.nn.Parameter(
        #     t.normal(mean=t.zeros(cfg["first_cheb_order"], self.in_channels, self.mid_channels).float(), std=cfg["init_std"]).float()
        # )
        # self.chev_conv_front.bias = t.nn.Parameter(
        #     t.normal(mean=t.zeros(self.mid_channels).float(), std=cfg["init_std"]).float()
        # )
        # # self.bn_front = t.nn.BatchNorm1d(self.mid_channels)
        # # self.dropout1 = t.nn.Dropout(p=cfg["drop_prob1"])
        # self.chev_conv_back = ChebConv(in_channels=self.mid_channels, out_channels=self.mid_channels, K=cfg["second_cheb_order"])
        # self.chev_conv_back.weight = t.nn.Parameter(
        #     t.normal(mean=t.zeros(cfg["second_cheb_order"], self.mid_channels, self.mid_channels).float(),
        #              std=cfg["init_std"]).float()
        # )
        # self.chev_conv_back.bias = t.nn.Parameter(
        #     t.normal(mean=t.zeros(self.mid_channels).float(), std=cfg["init_std"]).float()
        # )
        # self.bn_back = t.nn.BatchNorm1d(self.mid_channels)
        # self.dropout2 = t.nn.Dropout(p=cfg["drop_prob2"])

        # self.sftm = t.nn.Softmax(dim=-1)

    def forward(self, x, lap):
        front_out = self.cheb_conv_front(x, lap)
        front_out = F.relu(front_out)
        front_out = self.dropout1(front_out)
        # front_out = self.bn_front(front_out)
        back_out = self.cheb_conv_back(front_out, lap)
        back_out = F.relu(back_out)
        back_out = self.dropout1(back_out)
        # back_out = self.bn_back(back_out)
        front_feature = global_pooling(front_out)
        back_feature = global_pooling(back_out)
        feature = t.cat([front_feature, back_feature], dim=1)
        feature = self.dropout2(feature)
        feature = self.fc1(feature)
        feature = F.relu(feature)
        feature = self.dropout2(feature)
        feature = self.fc2(feature)
        # feature = t.nn.functional.log_softmax(feature, dim=-1)
        # feature = self.sftm(feature)
        return feature


if __name__ == '__main__':
    import yaml

    with open("../cfg/cfg.yml", 'r') as file:
        cfg = yaml.load(file)
    model = GraphGlobal(cfg)
    a = t.randn(2, 2)
    a = t.autograd.Variable(a, requires_grad=True)
    b = t.randn(2, 2)
    print(b)
    c = t.matmul(a, b)
    d = t.sum(c)
    d.backward()
    print(a.grad)

