#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

File Name : Trainer,py
File Description : Define the Trainer class for model training
Author : llw

"""
import os
import time

from sklearn.metrics import accuracy_score

from utils.tools import *


class Trainer:
    def __init__(self, model, criterion, scheduler, train_data, test_data, cfg, logger):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_data = test_data
        self.test_data = test_data
        self.max_epoch = int(cfg["max_epoch"])
        self.eval_step = int(cfg["eval_step"])
        self.loss = 0
        self.metric = 0
        self.best_score = 0
        self.cfg = cfg
        self.logger = logger

    def run(self):
        self.logger.info('-' * 20 + "New Training Starts" + '-' * 20)
        for epoch in range(self.max_epoch):
            tic = time.time()
            self.logger.info("Epoch {}/{}:".format(epoch, self.max_epoch))
            self.step()
            if (epoch + 1) % self.eval_step == 0:
                log_content, self.metric = evaluate(self.model, accuracy_score, self.test_data)
                self.logger.info(log_content)
                if self.best_score < self.metric:
                    self.best_score = self.metric
                    self.logger.info("Saving checkpoint to {}.".format(os.path.join(self.cfg["root_path"],
                                                                               self.cfg["checkpoint_path"],
                                                                               self.cfg["name"] +
                                                                               self.cfg["checkpoint_name"]))
                                )
                    t.save(
                        self.model.state_dict(),
                        os.path.join(self.cfg["root_path"], self.cfg["checkpoint_path"], self.cfg["name"] + self.cfg["checkpoint_name"])
                    )
            self.logger.info("loss: {}".format(self.loss))
            toc = time.time()
            self.logger.info("Elapsed time is {}s".format(toc - tic))

    def step(self):
        self.loss = 0
        for iteration, data in enumerate(tqdm(self.train_data)):
            self.scheduler.zero_grad()
            X, y, edge_index = data
            X = X.reshape(-1, 3).cuda()
            if X.shape[0] > self.cfg["num_points"]:
                continue
            y = y.reshape(1, ).cuda()
            edge_index = edge_index.reshape(2, -1).cuda()
            X_var = t.autograd.Variable(X).float()
            y_var = t.autograd.Variable(y).long()
            edge_index_var = t.autograd.Variable(edge_index).long()
            # input_var = [X_var, edge_index_var]
            # pred_var = parallel_model(self.model, input_var, self.cfg["gpu"], [self.cfg["gpu"]])
            pred_var = self.model(X_var, edge_index_var)
            loss = self.criterion(pred_var, y_var)
            loss.backward()
            self.scheduler.step()
            self.loss += loss.data
        self.loss = self.loss / len(self.train_data)


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler("test.log"))
    logger.info("test")