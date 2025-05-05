import os
import sys
sys.path.insert(1, os.path.realpath(os.path.curdir))

import hydra
import torch
import logging
import datetime

from omegaconf import DictConfig, OmegaConf

from main import get_train_fnc, get_test_fnc
from builder import *

OmegaConf.register_new_resolver("index", lambda x, idx: x[idx])
OmegaConf.register_new_resolver("datetime", lambda x: datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))


@hydra.main(version_base=None, config_path="../configs/")
def run(cfg):
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    # ensure cuda device is available
    assert torch.cuda.is_available(), "cuda device not found"

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # build experiment directory structure
    experiment_dir = build_experiment_dir(cfg)

    # build loggers
    logger, tensorboard_logger = build_loggers(experiment_dir, cfg)

    # build dataloaders
    dataloaders = build_dataloaders(cfg)

    # build model
    model, ckpt = build_model(cfg, experiment_dir, logger)

    if cfg.mode == "train":
        # build optimizer
        optimizer = build_optimizer(model, cfg, ckpt)

        # build scheduler
        scheduler = build_scheduler(optimizer, cfg.scheduler, ckpt)

        # train
        train_fnc = get_train_fnc(cfg.model.model_name)
        train_fnc(
            dataloaders, 
            model, 
            optimizer, 
            scheduler, 
            logger, 
            tensorboard_logger, 
            experiment_dir,
            cfg,
            start_epoch=ckpt.get("epoch", 0),
            best_metric_val=ckpt.get("best_metric_val", 1e10),
        )
    elif cfg.mode == "test":
        # test
        test_fnc = get_test_fnc(cfg.model.model_name)
        test_fnc(
            dataloaders, 
            model, 
            logger,
            tensorboard_logger,
            experiment_dir,
            cfg,
        )

    logging.shutdown()
    tensorboard_logger.close()

if __name__ == "__main__":
    run()