import torch
import numpy as np

from datasets.pretrain_dataset import PretrainDataset
from datasets.scanwcf_dataset import ScanWCFDataset
from datasets.mask3d_dataset import Mask3DDataset

from torch.utils.data import DataLoader


DATASETS = {
    "ShapeNet55": PretrainDataset,
    "ScanWCF": ScanWCFDataset,
    "Mask3D": Mask3DDataset,
}


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataset(cfg):
    dataset = DATASETS[cfg.dataset.dataset_name](cfg)

    return dataset


def get_dataloader(cfg):
    dataset = get_dataset(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size if cfg.split == "train" else 1,
        shuffle=(cfg.split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=(cfg.split == "train"),
        worker_init_fn=worker_init_fn,
    )

    return dataloader