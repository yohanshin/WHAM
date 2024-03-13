from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch

from configs import constants as _C
from .dataset3d import Dataset3D
from .dataset2d import Dataset2D
from ...utils.kp_utils import convert_kps
from smplx import SMPL


class Human36M(Dataset3D):
    def __init__(self, cfg, dset='train'):
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'human36m_{dset}_backbone.pth')
        parsed_data_path = parsed_data_path.replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(Human36M, self).__init__(cfg, parsed_data_path, dset=='train')

        self.has_3d = True
        self.has_smpl = False

        # Among 31 joints format, 14 common joints are avaialable
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
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'mpii3d_{dset}_backbone.pth')
        parsed_data_path = parsed_data_path.replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(MPII3D, self).__init__(cfg, parsed_data_path, dset=='train')
        
        self.has_3d = True
        self.has_smpl = False
        
        # Among 31 joints format, 14 common joints are avaialable
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
        self.mask[-14:] = 1

    @property
    def __name__(self, ):
        return 'MPII3D'
    
    def compute_3d_keypoints(self, index):
        return convert_kps(self.labels['joints3D'][index], 'spin', 'h36m'
            )[:, _C.KEYPOINTS.H36M_TO_J17].float()

class ThreeDPW(Dataset3D):
    def __init__(self, cfg, dset='train'):
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'3dpw_{dset}_backbone.pth')
        parsed_data_path = parsed_data_path.replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(ThreeDPW, self).__init__(cfg, parsed_data_path, dset=='train')
        
        self.has_3d = True
        self.has_smpl = True
        
        # Among 31 joints format, 14 common joints are avaialable
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
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'insta_{dset}_backbone.pth')
        parsed_data_path = parsed_data_path.replace('backbone', cfg.MODEL.BACKBONE.lower())
        super(InstaVariety, self).__init__(cfg, parsed_data_path, dset=='train')
        
        self.has_3d = False
        self.has_smpl = False
        
        # Among 31 joints format, 17 coco joints are avaialable
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
        self.mask[:17] = 1

    @property
    def __name__(self, ):
        return 'InstaVariety'