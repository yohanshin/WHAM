from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np

from .amass import AMASSDataset
from .videos import Human36M, ThreeDPW, MPII3D, InstaVariety
from lib.utils.data_utils import make_collate_fn


class DataFactory(torch.utils.data.Dataset):
    def __init__(self, cfg, train_stage='syn'):
        super(DataFactory, self).__init__()
        
        if train_stage == 'stage1':
            self.datasets = [AMASSDataset(cfg)]
            self.dataset_names = ['AMASS']
        elif train_stage == 'stage2':
            self.datasets = [
                AMASSDataset(cfg), ThreeDPW(cfg), 
                Human36M(cfg), MPII3D(cfg), InstaVariety(cfg)
            ]
            self.dataset_names = ['AMASS', '3DPW', 'Human36M', 'MPII3D', 'Insta']

        self._set_partition(cfg.DATASET.RATIO)
        self.lengths = [len(ds) for ds in self.datasets]

    @property
    def __name__(self, ):
        return 'MixedData'

    def prepare_video_batch(self):
        [ds.prepare_video_batch() for ds in self.datasets]
        self.lengths = [len(ds) for ds in self.datasets]

    def _set_partition(self, partition):
        self.partition = partition
        self.ratio = partition
        self.partition = np.array(self.partition).cumsum()
        self.partition /= self.partition[-1]

    def __len__(self):
        return max([l for l, r in zip(self.lengths, self.ratio) if r > 0])

    def __getitem__(self, index):
        # Get the dataset to sample from
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % self.lengths[i]]