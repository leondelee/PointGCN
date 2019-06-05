import os

class Parameters():
    def __init__(self):
        #self.pointNumber = 1024
        self.neighborNumber = 40
        self.outputClassN = 40
        self.pointNumber = 1024
        self.gcn_1_filter_n = 1000
        self.gcn_2_filter_n = 1000
        self.gcn_3_filter_n = 1000
        self.fc_1_n = 600
        self.chebyshev_1_Order = 4
        self.chebyshev_2_Order = 3
        self.keep_prob_1 = 0.9 #0.9 original
        self.keep_prob_2 = 0.55
        self.batchSize = 28
        self.testBatchSize = 1
        self.max_epoch = 260
        self.learningRate = 12e-4
        self.dataset = 'ModelNet40'
       # self.weighting_scheme = 'weighted'
        self.weighting_scheme = 'uniform'
        self.root_path = os.getcwd()
        self.modelDir = os.path.join(self.root_path, "global_pooling_model", "model")
        self.logDir = os.path.join(self.root_path, "global_pooling_model", "log")
        self.dataDir = os.path.join(self.root_path, "data")
        self.fileName = '0112_1024_40_cheby_4_3_modelnet40_max_var_first_second_layer'
        self.weight_scaler = 40  #50
        self.gpu = 0

if __name__ == '__main__':
    para = Parameters()
    print(para.root_path)