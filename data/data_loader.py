#! -*- encoding: utf-8 -*-
# Author: llw

import torch.utils.data as DT
from torch_geometric.nn import knn_graph

from data.factory import *
from utils.tools import *
from data.global_pooling_model.read_data import *


class ModelNet(DT.Dataset):
    def __init__(self, cfg, data_type='train', batch_index=0):
        super(ModelNet, self).__init__()
        self.num_points = cfg["num_points"]
        self.K = cfg["K"]
        self.pts_batch, self.label_batch = my_load_data(self.num_points, 'farthest_sampling', data_type, batch_index)
        self.lap_batch = my_prepareGraph(self.pts_batch, self.K, self.num_points, data_type, batch_index)
        self.lap_batch = self.lap_batch.tocsr()
        self.lap_batch = [lap for lap in self.lap_batch]

    def __getitem__(self, item):
        pts = self.pts_batch[item]
        label = self.label_batch[item]
        lap = self.lap_batch[item].todense()
        pts = t.tensor(pts).float()
        label = t.tensor(label).float()
        lap = t.tensor(lap.reshape(self.num_points, self.num_points)).float()
        return pts, label, lap

    def __len__(self):
        return len(self.pts_batch)

    def size(self):
        return self.__len__()


if __name__ == '__main__':
    cfg = get_cfg()
    ds = ModelNet(cfg=cfg, batch_index=0)
    test_data = DT.DataLoader(
       dataset=ds,
       batch_size=cfg["batch_size"],
       shuffle=False
    )
    for d in test_data:
        print(d)