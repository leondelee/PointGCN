#! -*- encoding: utf-8 -*-
# Author: llw
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'build'))

import pickle
import numpy as np
import torch.utils.data as DT
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_add

from data.factory import *
from utils.tools import *
import sdv


CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 
'picture', 'counter', 'desk', 'curtain', 'refridgerator', 
'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


class MyModelNet(DT.Dataset):
    def __init__(self, cfg, train=True, normalize=True):
        super(MyModelNet, self).__init__()
        self.num_pts = cfg['num_points']
        self.normalize = normalize
        self.raw_data = ModelNet(root=os.path.join(cfg['data_path'], 'ModelNet40'), name='40', train=train)

    def __getitem__(self, item):
        choice = np.random.choice(self.raw_data[item].pos.shape[0], self.raw_data[item].pos.shape[0])
        x, y = self.raw_data[item].pos[choice], self.raw_data[item].y
        if self.normalize:
            mean = t.mean(x, dim=0)
            std = t.sqrt(t.var(x, dim=0))
            x = (x - mean) / std
        return t.tensor(x).float(), t.tensor(y).long()

    def __len__(self):
        return len(self.raw_data)


class ObjectsDataset(DT.Dataset):
    def __init__(self, cfg, train=True):
        super(ObjectsDataset, self).__init__()
        self.num_pairs = int(cfg['num_pairs'])
        with open(os.path.join(cfg['data_path'], 'objects.pkl'), 'rb') as file:
            objects, lbs = pickle.load(file)
        if train:
            self.objects = objects[0:int(len(objects) * 0.9)]
            self.lbs = lbs[0:int(len(lbs) * 0.9)]
        else:
            self.objects = objects[int(len(objects) * 0.9):]
            self.lbs = lbs[int(len(lbs) * 0.9):]

    def __getitem__(self, item):
        idx_x, idx_y = np.random.randint(0, self / self.num_pairs, 2)
        obj_x, obj_y = self.objects[idx_x], self.objects[idx_y]
        idx_poi_x = np.random.randint(0, len(obj_x), 1)
        idx_poi_y = np.random.randint(0, len(obj_y), 1)
        feat_x = sdv.compute(obj_x, interest_point_idxs=list(idx_poi_x))
        feat_y = sdv.compute(obj_y, interest_point_idxs=list(idx_poi_y))
        lb = (self.lbs[idx_x] == self.lbs[idx_y])
        return t.tensor(feat_x).float().squeeze(), t.tensor(feat_y[0]).float().squeeze(), t.tensor(lb).long()

    def __len__(self):
        return len(self.objects) * self.num_pairs

    def __truediv__(self, other):
        return len(self.objects) * self.num_pairs / other


if __name__ == '__main__':
    cfg = get_cfg()
    ds = ObjectsDataset(cfg)
    print(ds[0])
    # with open('objects.pkl', 'rb') as file:
    #     objects, lbs = pickle.load(file)
    #     for idx, obj in enumerate(objects):
    #         # minx, miny, minz = np.min(obj[:, 0]), np.min(obj[:, 1]), np.min(obj[:, 2])
    #         # maxx, maxy, maxz = np.max(obj[:, 0]), np.max(obj[:, 1]), np.max(obj[:, 2])
    #         # print(minx, miny, minz)
    #         # print(maxx, maxy, maxz)
    #         # input()
    #         vis_pcd(obj)
    #         print(CLASS_LABELS[idx])
    #         input()



