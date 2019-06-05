# Author: llw
import torch as t

from utils.tools import get_cfg

cfg = get_cfg()


class ChebConv(t.nn.Module):
    def __init__(self, in_channels, out_channels, K, bias=True):
        assert K > 0
        super(ChebConv, self).__init__()
        self.weight = t.nn.Parameter(
            t.normal(mean=t.zeros(K, in_channels, out_channels).float(), std=cfg["init_cheb_std"] * t.ones(K, in_channels, out_channels).float())
        )
        if bias:
            self.bias = t.nn.Parameter(
                t.normal(mean=t.zeros(out_channels).float(), std=cfg["init_cheb_std"] * t.ones(out_channels).float())
            )
        else:
            self.register_parameter('bias', None)
        self.K = K

    def forward(self, pts, lap):
        cheb_k_minus_2 = pts
        out = t.matmul(cheb_k_minus_2, self.weight[0])
        if self.K == 1:
            return out
        cheb_k_minus_1 = t.matmul(lap, pts)
        out = out + t.matmul(cheb_k_minus_1, self.weight[1])
        if self.K == 2:
            return out
        for k in range(2, self.K):
            cheb_k = 2 * t.matmul(lap, cheb_k_minus_1) - cheb_k_minus_2
            cheb_k_minus_2, cheb_k_minus_1 = cheb_k_minus_1, cheb_k
            out = out + t.matmul(cheb_k_minus_1, self.weight[k])
        out = out + self.bias
        return out


if __name__ == '__main__':
    a = t.randn(2, 3, 4)
    print(a)
    b = t.randn(2, 4, 2)
    d = b
    print(t.matmul(a, b))

    d[0] = t.zeros(4, 2).float()
    c = t.matmul(a, d)
    print(c)

    d = b
    d[1] = t.zeros(4, 2).float()
    c = t.matmul(a, d)
    print(c)
    # import torch as t
    # import numpy as np
    # from torch_geometric.nn import ChebConv as SB
    # from torch_geometric.nn import knn_graph
    #
    # from data.global_pooling_model.read_data import *
    #
    # inputTrain, trainLabel, inputTest, testLabel = load_data(1024, 'farthest_sampling')
    #
    # cheb = ChebConv(3, 10, 3)
    # linear = t.nn.Linear(10240, 40)
    #
    # sb = SB(3, 10, 3)
    # sb.weight = cheb.weight
    # sb.bias = cheb.bias
    # # sb.weight = t.nn.Parameter(
    # #     t.normal(mean=t.zeros(3, 3, 10).float(), std=0.05).float()
    # # )
    # # sb.bias = t.nn.Parameter(
    # #     t.normal(mean=t.zeros(10).float(), std=0.05).float()
    # # )
    #
    # ls = t.nn.CrossEntropyLoss()
    #
    # pts = t.tensor(inputTrain[0][0])
    # lbl = t.tensor(trainLabel[0][0].reshape(1))
    # pts = t.autograd.Variable(pts)
    # lbl = t.autograd.Variable(lbl)
    # edge_index = knn_graph(pts, 40)
    # out_sb = sb(pts, edge_index).reshape([1, -1])
    # out_sb = linear(out_sb)
    # loss = ls(out_sb, lbl)
    # loss.backward()
    # print(sb.weight.grad)
    #
    # scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest, 40, 1024)
    # pts = pts.unsqueeze(0)
    # scaledLaplacianTrain = scaledLaplacianTrain[0].tocsr()
    # for sc in scaledLaplacianTrain:
    #     sc = sc.toarray().reshape(1, 1024, 1024)
    #     sc = t.tensor(sc).float()
    #     sc = t.autograd.Variable(sc)
    #     out_my = cheb(pts, sc).reshape(1, -1)
    #     out_my = linear(out_my)
    #     loss = ls(out_my, lbl)
    #     loss.backward()
    #     print(cheb.weight.grad)
    #     break
    # scaledLaplacianTrain = t.tensor(scaledLaplacianTrain).reshape(1, 1024, 1024)
    # scaledLaplacianTrain = t.autograd.Variable(scaledLaplacianTrain)
    # out_my = cheb(pts, scaledLaplacianTrain)
    # loss = ls(out_my, lbl)
    # loss.backward()
    # print(cheb.weight.grad)



