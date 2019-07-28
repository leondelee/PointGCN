#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.tfutils import get_current_tower_context, summary
from dataflow import  MyDataflow
import multiprocessing
import pickle
from model import Model
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from multiprocessing import cpu_count

if __name__ == '__main__':

    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    with open('objects.pkl', 'rb') as f:
        objects, labels = pickle.load(f)
        objects = np.asarray(objects)
        labels = np.asarray(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    train_idx, test_idx = next(iter(sss.split(objects, labels)))
    dataset_train = MultiProcessPrefetchData(MyDataflow(objects[train_idx], labels[train_idx]), cpu_count() // 2, cpu_count() // 2)
    dataset_test = MultiProcessPrefetchData(MyDataflow(objects[test_idx], labels[test_idx]), cpu_count() // 2, cpu_count() // 2)

    # How many iterations you want in each epoch.
    # This len(data) is the default value.
    steps_per_epoch = len(dataset_train)

    # get the config which contains everything necessary in a training
    config = TrainConfig(
        model=Model(),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),  # save the model after every epoch
            MaxSaver('accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            PeriodicTrigger(InferenceRunner(  # run inference(for validation) after every epoch
                dataset_test,  # the DataFlow instance used for validation
                [ScalarStats(['cost', 'accuracy'])]
            ), every_k_epochs=5)
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=300,
    )
    launch_train_with_config(config, SimpleTrainer())