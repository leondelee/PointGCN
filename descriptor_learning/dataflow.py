from tensorpack import *
import numpy as np
import random
import sys
sys.path.append('./src/build')
import sdv


class MyDataflow(DataFlow):
    def __init__(self, objects, labels):
        self.objects = np.asarray(objects)
        self.labels = np.asarray(labels)
        self.batch_size = 18

    def __len__(self):
        return len(self.objects) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            batch_x = []
            batch_y = []
            for i in range(self.batch_size):
                idx = np.where(self.labels == i)[0]
                # if len(idx) == 0:
                #     continue
                rnd_idx = np.random.choice(idx, size=2, replace=True)
                rnd_feature_x = sdv.compute(self.objects[rnd_idx[0]],
                                            interest_point_idxs=[np.random.randint(len(self.objects[rnd_idx[0]]))])[0]
                rnd_feature_y = sdv.compute(self.objects[rnd_idx[1]],
                                            interest_point_idxs=[np.random.randint(len(self.objects[rnd_idx[1]]))])[0]
                batch_x.append(rnd_feature_x)
                batch_y.append(rnd_feature_y)
            yield batch_x, batch_y
