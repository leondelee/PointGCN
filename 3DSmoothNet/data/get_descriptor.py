# Author: llw
import os
import pickle

import numpy as np
import open3d as o3d
from data.data_loader import MyModelNet
from utils.tools import get_cfg


def gen_ply_files(out_path, train=True):
    ds = MyModelNet(get_cfg(), train=train)
    ds_len = len(ds)
    print('output path is ', out_path)
    for i in range(ds_len):
        pts = ds[i][0].numpy()
        y = ds[i][1]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(os.path.join(out_path, '{}_{}.ply'.format(y, str(i))), pcd)
        print('Processing {}/{}.'.format(i + 1, ds_len))


def get_feature_collection(input_npz_folder):
    npz_file_list = os.listdir(input_npz_folder)
    npz_file_list = [os.path.join(input_npz_folder, f) for f in npz_file_list if 'npz' in f]
    lenght = len(npz_file_list)
    for idx, npz in enumerate(npz_file_list):
        array = np.load(npz)['data']
        cls = int(npz.split('/')[-1].split('_')[0])
        assert array.shape[0] == 1024
        for idxx, row in enumerate(array):
            with open('modelnet_descriptor_pkl/{}_{}.pkl'.format(idx, idxx), 'wb') as file:
                pickle.dump([cls, row], file)
                file.close()
        print('Processed {}/{}.'.format(idx + 1, lenght))


if __name__ == '__main__':
    # gen_ply_files('ply_file')
    get_feature_collection('./modelnet_descriptor/64_dim')
    # with open('descriptor.pkl', 'rb') as file:
    #     a = pickle.load(file)
    #     print(a)

