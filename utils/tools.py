# Author: llw
import os
import logging
from tqdm import tqdm

import yaml
import torch as t
import pptk


def show_point_clouds(pts, lbs):
    v = pptk.viewer(pts)
    v.attributes(lbs)


def normalize_point_cloud(pts):
    norm = pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2
    norm = t.sqrt(norm).reshape(-1, 1)
    pts = pts / norm
    return pts


def get_cfg():
    parent_path = os.path.dirname(__file__)
    cfg_path = os.path.join(parent_path, '..', 'cfg/cfg.yml')
    with open(cfg_path, "r") as file:
        cfg = yaml.load(file)
        file.close()
    return cfg


def get_logger(cfg):
    format = "%(asctime)s - %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format
    )
    logger = logging.getLogger(cfg["name"] + " - " + cfg["mode"])
    file_handler = logging.FileHandler(os.path.join(cfg["root_path"], cfg["log_path"], cfg["name"] + cfg["log_name"]))
    file_handler.setFormatter(logging.Formatter(format))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def get_checkpoints(cfg):
    checkpoint_path = os.path.join(cfg["root_path"], cfg["checkpoint_path"], cfg["name"] + cfg["checkpoint_name"])
    log_path = os.path.join(cfg["root_path"], cfg["log_path"], cfg["name"] + cfg["log_name"])
    if os.path.exists(checkpoint_path):
        action = input("Found checkpoint at {}.\nPlease type k(eep) or d(elete) or others to ignore.\n".format(checkpoint_path))
        if action == 'k':
            return checkpoint_path
        elif action == 'd':
            print("Deleting ", checkpoint_path)
            os.unlink(checkpoint_path)
            print("Deleting ", log_path)
            os.unlink(log_path)
    return None


def clean_logs_and_checkpoints(cfg):
    checkpoint_path = os.path.join(cfg["root_path"], cfg["checkpoint_path"], cfg["name"] + cfg["checkpoint_name"])
    log_path = os.path.join(cfg["root_path"], cfg["log_path"], cfg["name"] + cfg["log_name"])
    if os.path.exists(checkpoint_path):
        print("Deleting ", checkpoint_path)
        os.unlink(checkpoint_path)
    if os.path.exists(log_path):
        print("Deleting ", log_path)
        os.unlink(log_path)


def evaluate(model, metric, eval_data):
    model.eval()
    print("-------------------Evaluating model----------------------")
    log_content = ""
    res = 0
    cnt = 0
    for batch_data in tqdm(eval_data):
        X, y, lap = batch_data
        X = X.cuda()
        lap = lap.cuda()
        y = y.cpu()
        # edge_index = edge_index.reshape(2, -1).cuda()
        X_var = t.autograd.Variable(X).float()
        y_var = t.autograd.Variable(y).long()
        lap_var = t.autograd.Variable(lap).float()
        # input_var = [X_var, edge_index_var]
        # pred_var = parallel_model(self.model, input_var, self.cfg["gpu"], [self.cfg["gpu"]])
        out = model(X_var, lap_var)
        out = t.argmax(out, dim=-1).cpu()
        res += metric(out, y)
        cnt += 1
    res = res / cnt
    log_content += "average {metric_name} is {metric_value}.".format(metric_name=metric.__name__, metric_value=res)
    model.train()
    return log_content, res


def parallel_model(model, input, output_device=0, device_ids=None):
    if not device_ids:
        device_ids = [0, 1]
    pts, edge_index = input
    edge_index = edge_index.reshape([-1, 2])
    input = [pts, edge_index]
    replicas = t.nn.parallel.replicate(model, device_ids)
    inputs = t.nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    for idx, ipt in enumerate(inputs):
        inputs[idx][1] = inputs[idx][1].reshape([2, -1])
    outputs = t.nn.parallel.parallel_apply(replicas, inputs)
    return t.nn.parallel.gather(outputs, output_device)


if __name__ == '__main__':
    cfg = get_cfg()
    print(cfg)
