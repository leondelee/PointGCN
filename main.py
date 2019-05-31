# Author: llw
import argparse

import torch.utils.data as DT

from utils.trainer import Trainer
from utils.tools import *
from data.data_loader import ModelNet
from model.graph_nn import *


def arg_parse():
    cfg = get_cfg("cfg/cfg.yml")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=cfg["gpu"]
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=cfg["max_epoch"]
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=cfg["eval_step"]
    )
    parser.add_argument(
        "--name",
        type=str,
        default=cfg["name"]
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg["lr"]
    )
    args = parser.parse_args()
    for arg in vars(args):
        cfg[arg] = getattr(args, arg)
    return cfg


def train(model, criterion, scheduler, train_data, test_data, cfg, logger):
    trainer = Trainer(model, criterion, scheduler, train_data, test_data, cfg, logger)
    trainer.run()


if __name__ == '__main__':
    cfg = arg_parse()
    t.cuda.set_device(cfg["gpu"])
    assert t.cuda.current_device() == int(cfg["gpu"])

    # checkpoint = get_checkpoints(cfg)
    checkpoint = None
    model = GraphGlobal(cfg).cuda()
    if checkpoint:
        model.load_state_dict(t.load(checkpoint))

    if cfg["mode"] == "debug":
        clean_logs_and_checkpoints(cfg)

    logger = get_logger(cfg)
    log_content = "\nUsing Configuration:\n{\n"
    for key in cfg:
        log_content += "    {}: {}\n".format(key, cfg[key])
    logger.info(log_content + '}')

    criterion = t.nn.CrossEntropyLoss()

    optimizer = t.optim.Adam(
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = t.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg["lr_step"],
        gamma=cfg["lr_decay"]
    )

    train_data = DT.DataLoader(
        dataset=ModelNet(cfg, train=True),
        batch_size=cfg["batch_size"],
        shuffle=True
    )
    test_data = DT.DataLoader(
        dataset=ModelNet(cfg, train=False),
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    train(
        model=model,
        scheduler=optimizer,
        criterion=criterion,
        train_data=train_data,
        test_data=test_data,
        cfg=cfg,
        logger=logger
    )

    # evaluate(model, accuracy_score, test_data)