# Author: llw
import torch as t
from torch_geometric.nn import ChebConv as CBconv


class ChebConv(t.nn.Module):
    def __init__(self, in_channels, out_channels, K, cfg, bias=True):
        super(ChebConv, self).__init__()
        self.out_channels = out_channels
        self.conv_layer = CBconv(in_channels, out_channels, K, bias)
        self.conv_layer.weight = t.nn.Parameter(
            t.normal(mean=t.zeros(K, in_channels, out_channels).float(), std=cfg["init_cheb_std"] * t.ones(K, in_channels, out_channels).float())
        )
        if bias:
            self.conv_layer.bias = t.nn.Parameter(
                t.normal(mean=t.zeros(out_channels).float(), std=cfg["init_cheb_std"] * t.ones(out_channels).float())
            )
        else:
            self.conv_layer.register_parameter('bias', None)

    def forward(self, pts, edge_index):
        batch_size, num_pts, num_features = pts.shape
        out = t.rand(batch_size, num_pts, self.out_channels).cuda()
        for i in range(batch_size):
            out[i] = self.conv_layer(pts[i], edge_index[i])
        return out

# class ChebConv(t.nn.Module):
#     def __init__(self, in_channels, out_channels, K, bias=True):
#         assert K > 0
#         super(ChebConv, self).__init__()
#         self.weight = t.nn.Parameter(
#             t.normal(mean=t.zeros(K, in_channels, out_channels).float(), std=cfg["init_cheb_std"] * t.ones(K, in_channels, out_channels).float())
#         )
#         if bias:
#             self.bias = t.nn.Parameter(
#                 t.normal(mean=t.zeros(out_channels).float(), std=cfg["init_cheb_std"] * t.ones(out_channels).float())
#             )
#         else:
#             self.register_parameter('bias', None)
#         self.K = K
#
#     def forward(self, pts, lap):
#         cheb_k_minus_2 = pts
#         out = t.matmul(cheb_k_minus_2, self.weight[0])
#         if self.K == 1:
#             return out
#         cheb_k_minus_1 = t.matmul(lap, pts)
#         out = out + t.matmul(cheb_k_minus_1, self.weight[1])
#         if self.K == 2:
#             return out
#         for k in range(2, self.K):
#             cheb_k = 2 * t.matmul(lap, cheb_k_minus_1) - cheb_k_minus_2
#             cheb_k_minus_2, cheb_k_minus_1 = cheb_k_minus_1, cheb_k
#             out = out + t.matmul(cheb_k_minus_1, self.weight[k])
#         out = out + self.bias
#         return out




if __name__ == '__main__':
    ln_ = t.nn.SmoothL1Loss(reduce=False, size_average=False)
    ln = t.nn.SmoothL1Loss()
    a = t.randn(4, 5)
    b = t.randn(4, 5)
    loss_ = ln_(a, b)
    loss = ln(a, b)
    print(loss_, loss)



