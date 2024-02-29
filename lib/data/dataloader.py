from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from .datasets import EvalDataset, DataFactory
from ..utils.data_utils import make_collate_fn


def setup_eval_dataloader(cfg, data, split='test', backbone=None):
    if backbone is None:
        backbone = cfg.MODEL.BACKBONE
    
    dataset = EvalDataset(cfg, data, split, backbone)
    dloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=make_collate_fn()
    )
    return dloader


def setup_train_dataloader(cfg, ):
    n_workers = 0 if cfg.DEBUG else cfg.NUM_WORKERS
    
    train_dataset = DataFactory(cfg, cfg.TRAIN.STAGE)
    dloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=make_collate_fn()
    )
    return dloader


def setup_dloaders(cfg, dset='3dpw', split='val'):
    test_dloader = setup_eval_dataloader(cfg, dset, split, cfg.MODEL.BACKBONE)
    train_dloader = setup_train_dataloader(cfg)
    
    return train_dloader, test_dloader