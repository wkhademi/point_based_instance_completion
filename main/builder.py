import os
import torch
import logging
import torch.nn as nn

from models import MODELS
from datasets import get_dataloader
from utils.log_utils import log_model_parameters
from utils.scheduler_utils import (
    GradualWarmupScheduler, build_lambda_scheduler
)

from torch.utils.tensorboard import SummaryWriter


def build_experiment_dir(cfg):
    experiment_name = cfg.experiment_name
    experiment_dir = os.path.join("./experiments", experiment_name)

    if cfg.load_ckpt:
        assert os.path.exists(experiment_dir), \
            f"experiment {experiment_name} does not exist."

        return experiment_dir

    assert cfg.mode != "test", \
        "must load model checkpoint from existing experiment if running in test mode."

    # create new experiment (and other relevant dirs)
    os.makedirs(experiment_dir, exist_ok=True)
    ckpt_dir = os.path.join(experiment_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    return experiment_dir


def build_loggers(experiment_dir, cfg):
    # set logging info for file and stdout logging
    logger = logging.getLogger(cfg.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        os.path.join(experiment_dir, f"logs/{cfg.mode}_log.txt")
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("----------------------- Configuration -----------------------")
    logger.info(cfg)

    # set tensorboard logger
    tensorboard_logger = SummaryWriter(f"{experiment_dir}/logs")

    return logger, tensorboard_logger


def build_dataloaders(cfg):
    if cfg.mode == "train":
        train_dataloader = get_dataloader(cfg.train_dataloader)
        val_dataloader = get_dataloader(cfg.val_dataloader)

        return (train_dataloader, val_dataloader)
    elif cfg.mode == "test":
        test_dataloader = get_dataloader(cfg.test_dataloader)

        return test_dataloader
    else:
        raise NotImplementedError


def build_model(cfg, experiment_dir, logger):
    # build model
    model_cfg = cfg.model
    model_cls = MODELS[model_cfg.model_name]
    model = model_cls(model_cfg).cuda()

    # load model weights if checkpoint is provided
    ckpt = {}
    if cfg.load_ckpt:
        ckpt_pth = os.path.join(experiment_dir, f"ckpts/ckpt_{cfg.ckpt_version}.pth")
        assert os.path.exists(ckpt_pth), \
            f"no model checkpoint named {ckpt_pth} found."
        logger.info("Loading model weights...")
        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Model loaded successfully.")
    elif hasattr(cfg, "pretrained_path"):
        ckpt_pth = cfg.pretrained_path
        assert os.path.exists(ckpt_pth), \
            f"no model checkpoint named {ckpt_pth} found."
        logger.info("Loading pretrained object-level completion model weights...")
        pretrained_ckpt = torch.load(ckpt_pth)
        model.load_state_dict(pretrained_ckpt["model_state_dict"], strict=False)
        logger.info("Pretrained object-level completion model weights loaded successfully.")
    else:
        # log layers and parameter count of model
        log_model_parameters(model, logger)
        
    # use data parallelism if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.device_ids).cuda()

    return model, ckpt


def build_optimizer(model, cfg, ckpt):
    parameters = model.parameters()

    optimizer_cfg = cfg.optimizer

    # create optimizer
    if optimizer_cfg.type == "Adam":
        optimizer = torch.optim.Adam(
            parameters, 
            lr=optimizer_cfg.lr,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2), 
            weight_decay=optimizer_cfg.weight_decay,
        )
    if optimizer_cfg.type == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters, 
            lr=optimizer_cfg.lr,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2), 
            weight_decay=optimizer_cfg.weight_decay,
            amsgrad=optimizer_cfg.amsgrad,
        )
    elif optimizer_cfg.type == "SGD":
        optimizer = torch.optim.SGD(
            parameters, 
            lr=optimizer_cfg.lr,
            momentum=optimizer_cfg.momentum,
            dampening=optimizer_cfg.dampening,
            weight_decay=optimizer_cfg.weight_decay,
            nesterov=optimizer_cfg.nesterov,
        )

    # load optimizer settings from saved model checkpoint
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return optimizer


def build_scheduler(optimizer, scheduler_cfg, ckpt):
    # create scheduler
    if scheduler_cfg.type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_cfg.step_size,
            gamma=scheduler_cfg.gamma,
            last_epoch=scheduler_cfg.last_epoch,
        )
    elif scheduler_cfg.type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=scheduler_cfg.gamma,
            last_epoch=scheduler_cfg.last_epoch,
        )
    elif scheduler_cfg.type == "LambdaLR":
        scheduler = build_lambda_scheduler(
            optimizer, **scheduler_cfg.lambda_cfg
        )
    else:
        raise NotImplementedError

    # add warmup
    if scheduler_cfg.warmup.use:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=scheduler_cfg.warmup.multiplier,
            total_epoch=scheduler_cfg.warmup.total_epoch,
            after_scheduler=scheduler,
        )

    # load optimizer settings from saved model checkpoint
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # load scheduler settings from saved model checkpoint
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return scheduler
