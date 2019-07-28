#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

File Name : Trainer,py
File Description : Define the Trainer class for model training
Author : llw

"""
import os
import time

from sklearn.metrics import accuracy_score, mean_squared_error

from utils.tools import *


class Trainer:
    def __init__(self, cfg):
        self.model = cfg['trainer_config']['model']
        self.criterion = cfg['trainer_config']['criterion']
        self.scheduler = cfg['trainer_config']['scheduler']
        self.train_data = cfg['trainer_config']['train_data']
        self.logger = cfg['trainer_config']['logger']
        self.max_epoch = int(cfg["max_epoch"])
        self.eval_step = int(cfg["eval_step"])
        self.lr_step = int(cfg['lr_step'])
        self.lr_decay = float(cfg["lr_decay"])
        self.cfg = cfg
        self.loss = 0
        self.metric_value = 0
        self.best_score = 0
        self.iteration = 0

    def run(self):
        self.logger.info('-' * 20 + "New Training Starts" + '-' * 20)
        for epoch in range(self.max_epoch):
            tic = time.time()
            self.logger.info("Epoch {}/{}:".format(epoch, self.max_epoch))
            self.step()
            if (epoch + 1) % self.eval_step == 0:
                log_info = evaluate(self.cfg)
                log_content = "Average {} is {}.".format(log_info['metric_name'], log_info['value'])
                self.metric_value = log_info['value']
                self.logger.info(log_content)
                if self.best_score > self.metric_value:
                    self.best_score = self.metric_value
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
        self.iteration += 1
        if self.iteration % self.lr_step == 0:
            for group in self.scheduler.param_groups:
                group['lr'] *= self.lr_decay
                logger.info('Learning rate of {} is {}.'.format(group, group['lr']))
        cnt = 0
        self.model.train()
        for batch_data in tqdm(self.train_data):
            self.scheduler.zero_grad()
            output = self.model(batch_data)
            loss = self.criterion(output) # + feature_transform_reguliarzer(trans_feat) * 0.001
            loss.backward()
            self.scheduler.step()
            self.loss += loss.data
            cnt += 1
        self.loss = self.loss / cnt


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler("test.log"))
    logger.info("test")