from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from time import time

import torch

from configs import constants as _C
from .dataset3d import Dataset3D
from .dataset2d import Dataset2D
from ...utils.kp_utils import convert_kps
from smplx import SMPL

def get_mask(exp):
    mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
    if exp == 0:
        # Use COCO+Feet
        mask[:-14] = 1
    elif exp == 1:
        # Use COCO w.o face
        mask[:13] = 1
    elif exp == 2:
        # Use All w.o face + feet
        mask[:13] = 1
        mask[-14:] = 1
    elif exp == 3:
        # Use All
        mask[:] = 1
    return mask


class Human36M(Dataset3D):
    def __init__(self, cfg, dset='train'):
        t = time()
        fname = _C.PATHS.HUMAN36M_LABEL
        fname = fname.replace('dset', dset).replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(Human36M, self).__init__(cfg, fname, dset=='train')

        self.has_3d = True
        self.has_smpl = cfg.DATASET.USE_NEURAL_ANNOT
        if cfg.DATASET.USE_NEURAL_ANNOT:
            self.mask = get_mask(cfg.TRIAL.DATASET3D_MASK)
        else:
            self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
            self.mask[-14:] = 1

    @property
    def __name__(self, ):
        return 'Human36M'

    def compute_3d_keypoints(self, index):
        return convert_kps(self.labels['joints3D'][index], 'spin', 'h36m'
            )[:, _C.KEYPOINTS.H36M_TO_J14].float()

class MPII3D(Dataset3D):
    def __init__(self, cfg, dset='train'):
        t = time()
        fname = _C.PATHS.MPII3D_LABEL
        fname = fname.replace('dset', dset).replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(MPII3D, self).__init__(cfg, fname, dset=='train')
        
        self.has_smpl = cfg.DATASET.USE_NEURAL_ANNOT
        if cfg.DATASET.USE_NEURAL_ANNOT:
            self.mask = get_mask(cfg.TRIAL.DATASET3D_MASK)
        else:
            self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
            self.mask[-14:] = 1

    @property
    def __name__(self, ):
        return 'MPII3D'
    
    def compute_3d_keypoints(self, index):
        return convert_kps(self.labels['joints3D'][index], 'spin', 'h36m'
            )[:, _C.KEYPOINTS.H36M_TO_J17].float()

class RICH(Dataset3D):
    def __init__(self, cfg, dset='test'):
        fname = _C.PATHS.RICH_LABEL
        fname = fname.replace('dset', dset).replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(RICH, self).__init__(cfg, fname, dset=='train')
        
        self.has_smpl = True
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
    
    @property
    def __name__(self, ):
        return 'RICH'
    
    def compute_3d_keypoints(self, index):
        return self.labels['joints3D'][index]
    
class EMDB(Dataset3D):
    def __init__(self, cfg, dset='1'):
        fname = _C.PATHS.EMDB_LABEL.replace('emdb', 'emdb' + dset)
        fname = fname.replace('dset', 'test').replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(EMDB, self).__init__(cfg, fname, dset=='train')
        
        self.has_smpl = True
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
    
    @property
    def __name__(self, ):
        return 'EMDB'
    
    def compute_3d_keypoints(self, index):
        return self.labels['joints3D'][index]


class ThreeDPW(Dataset3D):
    def __init__(self, cfg, dset='train'):
        fname = _C.PATHS.THREEDPW_LABEL
        fname = fname.replace('dset', dset).replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(ThreeDPW, self).__init__(cfg, fname, dset=='train')
        
        self.has_smpl = True
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
        self.mask[:-14] = 1
        
        self.smpl_gender = {
            0: SMPL(_C.BMODEL.FLDR, gender='male', num_betas=10),
            1: SMPL(_C.BMODEL.FLDR, gender='female', num_betas=10)
        }

    @property
    def __name__(self, ):
        return 'ThreeDPW'
    
    def compute_3d_keypoints(self, index):
        return self.labels['joints3D'][index]
    
    
class InstaVariety(Dataset2D):
    def __init__(self, cfg, dset='train'):
        t = time()
        fname = _C.PATHS.INSTA_LABEL
        fname = fname.replace('dset', dset).replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(InstaVariety, self).__init__(cfg, fname, dset=='train')
        
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
        self.mask[:17] = 1

    @property
    def __name__(self, ):
        return 'InstaVariety'