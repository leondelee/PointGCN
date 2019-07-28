# Author: llw
import os
import sys
this_path = os.path.dirname(os.path.abspath(__file__))   # data
ext_path = os.path.join(this_path, '..', 'ext')
sys.path.insert(0, ext_path)

from ext.SIS.code.lib.datasets.dataloader import *
from ext.SIS.code.lib.datasets.dataset import *


def get_scannet_data(mode):
    if mode == 'train':
        return dataset_train
    # elif mode == 'val':
    #     return dataset_val
    # else:
    #     return dataset_test


if __name__ == '__main__':
    dataset_train = dataset_train
    # dataloader_train = get_dataloader(dataset_train, batch_size=2, num_workers=1)
    for i in range(len(dataset_train)):
        data = dataset_train[0]['data']
        print(data)
        if i == 1:
            break
