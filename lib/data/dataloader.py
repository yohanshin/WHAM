from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from lib.utils.data_utils import make_collate_fn
from ._dataset_eval import EvalDataset

import torch

def setup_eval_dataloader(cfg, data, split='test', backbone=None):
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