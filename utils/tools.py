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


def get_cfg(args):
    name = args.name
    parent_path = os.path.dirname(__file__)
    cfg_path = os.path.join(parent_path, '..', 'cfg/{}.yml'.format(name))
    with open(cfg_path, "r") as file:
        cfg = yaml.load(file)
        file.close()
    for arg in vars(args):
        if getattr(args, arg) != '-1':
            cfg[arg] = getattr(args, arg)
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


def evaluate(cfg):
    # model.eval()
    model = cfg['trainer_config']['model']
    test_data = cfg['trainer_config']['test_data']
    metric = cfg['trainer_config']['metric']
    print("-------------------Evaluating model----------------------")
    res = 0
    cnt = 0
    for batch_data in tqdm(test_data):
        output = model(batch_data)
        res += metric(output)
        cnt += 1
    res = res / cnt
    model.train()
    log_info = dict(
        metric_name=metric.__name__,
        value=res
    )
    return log_info


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
    import torch as t
    from sklearn.metrics import mean_squared_error
    a = t.tensor([[1, 2.1], [2, 3]])
    b = t.tensor([[1, 2], [1, 2]])
    a = t.autograd.Variable(a, requires_grad=True)
    print(t.detach(a))
