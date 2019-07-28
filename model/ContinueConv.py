# Author: llw
import torch as t


class ContinuousKernel(t.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContinuousKernel, self).__init__()
        self.layers = t.nn.Sequential(
            t.nn.Linear(in_features=in_channels, out_features=64),
            t.nn.ReLU(),
            t.nn.Linear(in_features=64, out_features=32),
            t.nn.ReLU(),
            t.nn.Linear(in_features=32, out_features=out_channels)
        )

    def forward(self, x):
        return self.layers(x.float())


class ContinuousConv(t.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContinuousConv, self).__init__()
        self.kernel = ContinuousKernel(in_channels, out_channels)

    def forward(self, x):
        pass

if __name__ == '__main__':
    cc = ContinuousKernel(3, 5)
    a = t.ones(2, 4, 3)
    print(a)
    b = cc(a)
    print(b)
