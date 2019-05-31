# Author: llw
import torch.utils.data as DT
from torch_geometric.nn import knn_graph

from data.factory import *


class ModelNet(DT.Dataset):
    def __init__(self, cfg, train=True):
        super(ModelNet, self).__init__()
        self.cfg = cfg
        if train:
            self.data = get_modelnet_data(cfg, train=train)
        else:
            self.data = get_modelnet_data(cfg, train=train)

    def __getitem__(self, item):
        choice = t.randperm(self.data[item].pos.shape[0])
        num_points = self.data[item].pos.shape[0]
        pos = self.data[item].pos[choice]
        y = self.data[item].y
        edge_index = knn_graph(pos[0:num_points], k=self.cfg["K"])
        return pos[0:num_points], y, edge_index

    def __len__(self):
        return 309

    def size(self):
        return self.__len__()


if __name__ == '__main__':
    with open("../cfg/cfg.yml", 'r') as file:
        cfg = yaml.load(file)
    ds = ModelNet(cfg, False)
    print(ds[0])
    dl = DT.DataLoader(ds, batch_size=1, shuffle=True, drop_last=True)
    for d in dl:
        print(d[0].shape)
    # a = t.randn(4,)
    # choice = t.randperm(4)
    # print(a[0:2])
    # print(a.shape)
    # print(a[choice])
