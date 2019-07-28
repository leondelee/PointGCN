import torch as t
import numpy as np


class ConvBlock(t.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, drop_prob=0.0):
        super(ConvBlock, self).__init__()
        self.conv_layer = t.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn_layer = t.nn.BatchNorm3d(out_channels)
        self.relu_layer = t.nn.ReLU()
        self.dropout_layer = t.nn.Dropout(p=drop_prob)

    def forward(self, input):
        out = self.conv_layer(input)
        out = self.bn_layer(out)
        out = self.relu_layer(out)
        out = self.dropout_layer(out)
        return out


class OutBlock(t.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(OutBlock, self).__init__()
        self.conv_layer = t.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn_layer = t.nn.BatchNorm3d(out_channels)

    def forward(self, input):
        out = self.conv_layer(input)
        out = self.bn_layer(out)
        return out.reshape(-1)


class SmoothNet3D(t.nn.Module):
    def __init__(self, cfg):
        super(SmoothNet3D, self).__init__()
        self.input_dim = 4096
        self.net_blocks = t.nn.Sequential(
            ConvBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, drop_prob=0.3),
            OutBlock(in_channels=128, out_channels=32, kernel_size=8, stride=1)
        )

    def forward(self, input):
        x, y, lb = input
        cbrt = np.cbrt(self.input_dim).astype(np.int32)
        batch_size = x.shape[0]
        x_var = t.autograd.Variable(x).cuda().reshape(-1, 1, cbrt, cbrt, cbrt)
        y_var = t.autograd.Variable(x).cuda().reshape(-1, 1, cbrt, cbrt, cbrt)
        x_output = self.net_blocks(x_var).reshape(batch_size, -1)
        y_output = self.net_blocks(y_var).reshape(batch_size, -1)
        return [x_output, y_output, lb]


class DescLoss(t.nn.Module):
    def __init__(self):
        super(DescLoss, self).__init__()

    def forward(self, input):
        x, y, lb = input
        xy_dist = t.norm(x - y, dim=-1)
        same_mask = (lb).float().cuda()
        diff_mask = (1 - lb).float().cuda()
        loss = t.log(1 + t.exp(xy_dist * same_mask - xy_dist * diff_mask)).squeeze()
        return t.mean(loss, dim=0)


class DescMetric(t.nn.Module):
    def __init__(self, input_dim=4096):
        super(DescMetric, self).__init__()
        self.desc_loss = DescLoss()
        self.__name__ = self.name()

    def name(self):
        # super(DescMetric, self).__name__()
        # return 'desc metric'
        return self.__class__.__name__

    def forward(self, input):
        return self.desc_loss(input).detach().cpu()


if __name__ == '__main__':
    metric = DescMetric()
    print(metric.__name__)