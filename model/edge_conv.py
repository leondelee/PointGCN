# Author: llw
import torch as t
from torch_geometric.nn import EdgeConv, knn_graph


class EdgeConvBlock(t.nn.Module):
    def __init__(self, nn, aggr):
        super(EdgeConvBlock, self).__init__()
        self.out_channels = nn[0].out_features
        self.edge_conv_layer = EdgeConv(nn, aggr)

    def forward(self, pts, edge_index):
        batch_size, num_pts, num_features = pts.shape
        out = t.randn(batch_size, num_pts, self.out_channels)
        if pts.is_cuda:
            out = out.cuda()
        for batch in range(batch_size):
            out[batch] = self.edge_conv_layer(pts[batch], edge_index[batch])
        return out


class KnnBlock(t.nn.Module):
    def __init__(self, K):
        super(KnnBlock, self).__init__()
        self.K = K

    def forward(self, pts):
        batch_size, num_pts, _ = pts.shape
        out = []
        flag = pts.is_cuda
        for batch in range(batch_size):
            edge_index = knn_graph(pts[batch], self.K)
            if flag:
                edge_index = edge_index.cuda()
            out.append(edge_index)
        return out


class DynamicEdgeConv(t.nn.Module):
    def __init__(self, cfg, aggr='max'):
        in_channels = cfg['in_channels']
        out_channels = cfg['out_channels']
        super(DynamicEdgeConv, self).__init__()
        self.K = cfg['K']
        self.edge_conv1 = EdgeConvBlock(t.nn.Sequential(
            t.nn.Linear(in_channels * 2, 64),
            t.nn.ReLU(),
            t.nn.BatchNorm1d(64)
        ), aggr)
        self.edge_conv2 = EdgeConvBlock(t.nn.Sequential(
            t.nn.Linear(64 * 2, 64),
            t.nn.ReLU(),
            t.nn.BatchNorm1d(64)
        ), aggr)
        self.edge_conv3 = EdgeConvBlock(t.nn.Sequential(
            t.nn.Linear(64 * 2, 128),
            t.nn.ReLU(),
            t.nn.BatchNorm1d(128)
        ), aggr)
        self.edge_conv4 = EdgeConvBlock(t.nn.Sequential(
            t.nn.Linear(128 * 2, 256),
            t.nn.ReLU(),
            t.nn.BatchNorm1d(256)
        ), aggr)
        self.knn1 = KnnBlock(self.K)
        self.knn2 = KnnBlock(self.K)
        self.knn3 = KnnBlock(self.K)
        self.fc = t.nn.Sequential(
            t.nn.Linear(512, 256),
            t.nn.ReLU(),
            # t.nn.Dropout(p=cfg['drop_prob']),
            t.nn.Linear(256, out_channels)
        )

    def forward(self, pts0, edge_index0):
        pts1 = self.edge_conv1(pts0, edge_index0)
        edge_index1 = self.knn1(pts1)
        pts2 = self.edge_conv2(pts1, edge_index1)
        edge_index2 = self.knn2(pts2)
        pts3 = self.edge_conv3(pts2, edge_index2)
        edge_index3 = self.knn3(pts3)
        pts4 = self.edge_conv4(pts3, edge_index3)
        out = t.cat([pts1, pts2, pts3, pts4], dim=-1)
        out = t.max(out, 1).values
        out = self.fc(out)
        return out


if __name__ == '__main__':
    from utils.tools import get_cfg
    cfg = get_cfg()
    a = t.randn(2, 50, 3)
    edg = knn_graph(a[0], cfg['K']).unsqueeze(0)
    edg = t.cat([edg, edg], dim=0)
    print(edg.shape)
    model = DynamicEdgeConv(3, 10, cfg)
    b = model(a, edg)
    print(b)