#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:25:59 2017

@author: yingxuezhang
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import global_pooling_model.utils as utils
import scipy
# from utils import adjacency, scaled_laplacian
from global_pooling_model.utils import adjacency, scaled_laplacian
import numpy as np
from scipy.spatial import cKDTree
import pickle
from global_pooling_model.Parameters import Parameters

# import scipy
# import utils
# from utils import adjacency, scaled_laplacian
# import numpy as np
# from scipy.spatial import cKDTree
# import pickle
# from Parameters import Parameters

para = Parameters()
def farthestSampling(file_names, NUM_POINT):
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range (len(file_indexs)):
        # print(file_names)
        current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label) 
        current_label= np.int_(current_label)
        inputData.update({index : current_data})
        inputLabel.update({index : current_label})
    return inputData, inputLabel


def uniformSampling(file_names, NUM_POINT):
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range (len(file_indexs)):
        current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
        current_label = np.squeeze(current_label) 
        current_label= np.int_(current_label)
        output = np.zeros((len(current_data), NUM_POINT, 3))
        for i,object_xyz in enumerate (current_data):
            samples_index=np.random.choice(2048, NUM_POINT, replace=False)
            output[i] = object_xyz[samples_index]
        inputData.update({index : output})
        inputLabel.update({index : current_label})
    return inputData, inputLabel


# ModelNet40 official train/test split
# def load_data(NUM_POINT, sampleType):
#     # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     # BASE_DIR= os.path.abspath(os.path.dirname(os.getcwd()))
#     # BASE_DIR = para.root_path
#     BASE_DIR = '/home/liangwei/3D/graph'
#
#     #print BASE_DIR
#     # BASE_DIR = '/raid60/yingxue.zhang2/ICASSP_code/'
#     TRAIN_FILES = utils.getDataFiles(\
#         os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
#     TEST_FILES = utils.getDataFiles(\
#         os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
#     TRAIN_FILES = [os.path.join(BASE_DIR, pth) for pth in TRAIN_FILES]
#     TEST_FILES = [os.path.join(BASE_DIR, pth) for pth in TEST_FILES]
#     #np.random.shuffle(train_file_idxs)
#     if sampleType == 'farthest_sampling':
#         inputTrainFarthest, inputTrainLabel = farthestSampling(TRAIN_FILES, NUM_POINT)
#         inputTestFathest, inputTestLabel = farthestSampling(TEST_FILES, NUM_POINT)
#         return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel
#
#     elif sampleType == 'uniform_sampling':
#         inputTrainFarthest, inputTrainLabel = uniformSampling(TRAIN_FILES, NUM_POINT)
#         inputTestFathest, inputTestLabel = uniformSampling(TEST_FILES, NUM_POINT)
#
#         return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel

def load_data(NUM_POINT, sampleType):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR= os.path.abspath(os.path.dirname(os.getcwd()))
    # BASE_DIR = para.root_path
    # BASE_DIR = para.root_path

    # print BASE_DIR
    # BASE_DIR = '/raid60/yingxue.zhang2/ICASSP_code/'
    TRAIN_FILES = utils.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    TEST_FILES = utils.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    TRAIN_FILES = [os.path.join(BASE_DIR, pth) for pth in TRAIN_FILES]
    TEST_FILES = [os.path.join(BASE_DIR, pth) for pth in TEST_FILES]
    # np.random.shuffle(train_file_idxs)
    if sampleType == 'farthest_sampling':
        inputTrainFarthest, inputTrainLabel = farthestSampling(TRAIN_FILES, NUM_POINT)
        inputTestFathest, inputTestLabel = farthestSampling(TEST_FILES, NUM_POINT)
        return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel

    elif sampleType == 'uniform_sampling':
        inputTrainFarthest, inputTrainLabel = uniformSampling(TRAIN_FILES, NUM_POINT)
        inputTestFathest, inputTestLabel = uniformSampling(TEST_FILES, NUM_POINT)

        return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel


#generate graph structure and store in the system
def prepareGraph(inputData, neighborNumber, pointNumber, dataType):
    scaledLaplacianDict = dict()
    baseDir = os.path.dirname(os.path.abspath(__file__))
    # baseDir ='/raid60/yingxue.zhang2/ICASSP_code'
    #baseDir= os.path.abspath(os.path.dirname(os.getcwd()))
    if para.dataset == 'ModelNet40':
        fileDir =  baseDir+ '/graph/' + dataType+'_pn_'+str(pointNumber)+'_nn_'+str(neighborNumber)
    elif para.dataset == 'ModelNet10':
        fileDir =  baseDir+ '/graph_ModelNet10/' + dataType+'_pn_'+str(pointNumber)+'_nn_'+str(neighborNumber)
    else:
        print("Please enter a valid dataset")
        
    if (not os.path.isdir(fileDir)):
        print("calculating the graph data")
        os.makedirs(fileDir)
        for batchIndex in range(len(inputData)):
            batchInput = inputData[batchIndex]
            print(batchIndex)
            for i in range(len(batchInput)):
                # print("dealing {}/{}".format(i + 1, len(batchInput)))
                pcCoordinates = batchInput[i]
                tree = cKDTree(pcCoordinates)
                dd, ii = tree.query(pcCoordinates, k = neighborNumber)
                A = adjacency(dd, ii)
                scaledLaplacian = scaled_laplacian(A)
                flattenLaplacian = scaledLaplacian.tolil().reshape((1, pointNumber*pointNumber))
                if i ==0:
                    batchFlattenLaplacian = flattenLaplacian
                else:
                    batchFlattenLaplacian = scipy.sparse.vstack([batchFlattenLaplacian, flattenLaplacian])
            scaledLaplacianDict.update({batchIndex: batchFlattenLaplacian})
            with open(fileDir+'/batchGraph_'+str(batchIndex), 'wb') as handle:
                pickle.dump(batchFlattenLaplacian, handle)
            print("Saving the graph data batch"+str(batchIndex))
        
    else:
        print("Loading the graph data from "+dataType+'Data')
        scaledLaplacianDict = loadGraph(inputData, neighborNumber, pointNumber, fileDir)
    return scaledLaplacianDict


def loadGraph(inputData, neighborNumber, pointNumber, fileDir):
    scaledLaplacianDict = dict()
    for batchIndex in range(len(inputData)):
        batchDataDir = fileDir+'/batchGraph_'+str(batchIndex)
        with open(batchDataDir, 'rb') as handle:
            batchGraph = pickle.load(handle)
        scaledLaplacianDict.update({batchIndex: batchGraph })
        print("Finish loading batch_"+str(batchIndex))
    return scaledLaplacianDict
        
                        
def prepareData(inputTrain, inputTest, neighborNumber, pointNumber):
    scaledLaplacianTrain = prepareGraph(inputTrain, neighborNumber, pointNumber, 'train')
    scaledLaplacianTest = prepareGraph(inputTest, neighborNumber, pointNumber, 'test')
    return scaledLaplacianTrain, scaledLaplacianTest


def my_prepareData(pts, neighborNumber, pointNumber, data_type, index):
    return my_prepareGraph(pts, neighborNumber, pointNumber, data_type, index)


def my_prepareGraph(pts, neighborNumber, pointNumber, dataType, index):
    baseDir = os.path.dirname(os.path.abspath(__file__))
    # baseDir ='/raid60/yingxue.zhang2/ICASSP_code'
    # baseDir= os.path.abspath(os.path.dirname(os.getcwd()))
    if para.dataset == 'ModelNet40':
        fileDir = baseDir + '/graph/' + dataType + '_pn_' + str(pointNumber) + '_nn_' + str(neighborNumber)
    elif para.dataset == 'ModelNet10':
        fileDir = baseDir + '/graph_ModelNet10/' + dataType + '_pn_' + str(pointNumber) + '_nn_' + str(neighborNumber)
    else:
        print("Please enter a valid dataset")
    print("Loading the graph data from " + dataType + 'Data')
    scaledLaplacian = my_loadGraph(pts, neighborNumber, pointNumber, fileDir, index)
    return scaledLaplacian


def my_loadGraph(pts, neighborNumber, pointNumber, fileDir, batchIndex):
    batchDataDir = fileDir + '/batchGraph_' + str(batchIndex)
    with open(batchDataDir, 'rb') as handle:
        batchGraph = pickle.load(handle)
    print("Finish loading batch_" + str(batchIndex))
    return batchGraph


def my_load_data(NUM_POINT, sampleType, data_type, index):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # BASE_DIR= os.path.abspath(os.path.dirname(os.getcwd()))
    # BASE_DIR = para.root_path
    BASE_DIR = '/home/liangwei/3D/graph'

    # print BASE_DIR
    # BASE_DIR = '/raid60/yingxue.zhang2/ICASSP_code/'
    # TRAIN_FILES = utils.getDataFiles( \
    #     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    # TEST_FILES = utils.getDataFiles( \
    #     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    # TRAIN_FILES = [os.path.join(BASE_DIR, pth) for pth in TRAIN_FILES]
    # TEST_FILES = [os.path.join(BASE_DIR, pth) for pth in TEST_FILES]
    FILES = utils.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/{}_files.txt'.format(data_type)))
    FILES = [os.path.join(BASE_DIR, pth) for pth in FILES]
    # np.random.shuffle(train_file_idxs)
    if sampleType == 'farthest_sampling':
        pts, labels = my_farthestSampling(FILES, NUM_POINT, index)
        return pts, labels

    # elif sampleType == 'uniform_sampling':
    #     pts, labels = uniformSampling(FILES, NUM_POINT)
    #     return pts, labels


def my_farthestSampling(file_names, NUM_POINT, index):
    file_indexs = np.arange(0, len(file_names))

        # print(file_names)
    current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label) 
    current_label= np.int_(current_label)
    return current_data, current_label


def my_uniformSampling(file_names, NUM_POINT, index):
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range (len(file_indexs)):
        current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
        current_label = np.squeeze(current_label) 
        current_label= np.int_(current_label)
        output = np.zeros((len(current_data), NUM_POINT, 3))
        for i,object_xyz in enumerate (current_data):
            samples_index=np.random.choice(2048, NUM_POINT, replace=False)
            output[i] = object_xyz[samples_index]
        inputData.update({index : output})
        inputLabel.update({index : current_label})
    return inputData, inputLabel


