# Author: llw
import torch as t


class DescCls(t.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DescCls, self).__init__()
        self.net_block = t.nn.Sequential(
            t.nn.Linear(in_channels, 512),
            t.nn.ReLU(),
            t.nn.Linear(512, 256),
            t.nn.ReLU(),
            t.nn.Linear(256, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 64),
            t.nn.ReLU(),
            t.nn.Linear(64, out_channels)
        )

    def forward(self, x):
        return self.net_block(x)