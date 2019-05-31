# Author: llw
import os
import yaml

import torch as t
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as Ts


def get_modelnet_data(cfg, train=True):
    if train:
        print('-' * 20 + "Loading ModelNet Training Data" + '-' * 20)
        model_net_data = ModelNet(
            root=os.path.join(cfg["root_path"], cfg["data_path"], "modelnet_train"),
            name="10",
            train=True
        )
    else:
        print('-' * 20 + "Loading ModelNet Testing Data" + '-' * 20)
        model_net_data = ModelNet(
            root=os.path.join(cfg["root_path"], cfg["data_path"], "modelnet_test"),
            name="10",
            train=False
        )

    return model_net_data


if __name__ == '__main__':
    from utils.tools import show_point_clouds
    with open("../cfg/cfg.yml", 'r') as file:
        cfg = yaml.load(file)
    train = get_modelnet_data(cfg)
    print(train[0].pos)
    # from torch_geometric.nn import knn_graph
    # x = t.randn(2000, 1000)
    # knn = knn_graph(x, 6)
    # print(knn)

