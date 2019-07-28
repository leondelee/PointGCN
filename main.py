import argparse

import torch
import torch.utils.data as DT
from sklearn.metrics import accuracy_score

from utils.trainer import Trainer
from utils.tools import *
from data.data_loader import ObjectsDataset as DataSet
from model.smoothnet3d import SmoothNet3D as Model
from model.smoothnet3d import DescLoss as Loss
from model.smoothnet3d import DescMetric as Metric


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        default='demo'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='-1'
    )
    parser.add_argument(
        '--batch_size',
        type=str,
        default='-1'
    )
    parser.add_argument(
        '--num_pairs',
        type=str,
        default='-1'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='train'
    )

    args = parser.parse_args()
    cfg = get_cfg(args)
    return cfg


def train(config):
    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    cfg = arg_parse()
    t.cuda.set_device(int(cfg["gpu"]))
    assert t.cuda.current_device() == int(cfg["gpu"])

    # checkpoint = get_checkpoints(cfg)
    checkpoint = None
    # model = GraphGlobal(cfg).cuda()
    model = Model(cfg).cuda()
    if checkpoint:
        model.load_state_dict(t.load(checkpoint))

    if cfg["mode"] == "debug":
        clean_logs_and_checkpoints(cfg)

    logger = get_logger(cfg)
    log_content = "\nUsing Configuration:\n{\n"
    for key in cfg:
        log_content += "    {}: {}\n".format(key, cfg[key])
    logger.info(log_content + '}')

    criterion = Loss()
    metric = Metric()
    # criterion = t.nn.NLLLoss()

    optimizer = t.optim.Adam(
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    train_data = DT.DataLoader(
        dataset=DataSet(cfg, True),
        batch_size=int(cfg["batch_size"]),
        shuffle=True
    )

    test_data = DT.DataLoader(
        dataset=DataSet(cfg, False),
        batch_size=int(cfg["batch_size"]),
        shuffle=True
    )
    cfg['trainer_config'] = dict(
        model=model,
        criterion=criterion,
        metric=metric,
        logger=logger,
        scheduler=optimizer,
        train_data=train_data,
        test_data=test_data
    )
    if cfg['task'] == 'train':
        train(cfg)
    else:
        evaluate(cfg)




