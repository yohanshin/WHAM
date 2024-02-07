from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
import joblib
import numpy as np

from configs import constants as _C
import lib.utils.data_utils as d_utils
from lib.utils import transforms
from lib.utils.kp_utils import root_centering
from lib.data._dataset import BaseDataset

FPS = 30
class EvalDataset(BaseDataset):
    def __init__(self, cfg, data, split, backbone):
        super(EvalDataset, self).__init__(cfg, False)
        
        self.prefix = ''
        self.data = data
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'{data}_{split}_{backbone}.pth')
        self.labels = joblib.load(parsed_data_path)

    def load_data(self, index, flip=False):
        if flip:
            self.prefix = 'flipped_'
        else:
            self.prefix = ''
        
        target = self.__getitem__(index)
        for key, val in target.items():
            if isinstance(val, torch.Tensor):
                target[key] = val.unsqueeze(0)
        return target

    def __getitem__(self, index):
        target = {}
        target = self.get_data(index)
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target

    def __len__(self):
        return len(self.labels['kp2d'])

    def prepare_labels(self, index, target):
        # Ground truth SMPL parameters
        target['pose'] = transforms.axis_angle_to_matrix(self.labels['pose'][index].reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][index]
        target['gender'] = self.labels['gender'][index]
        
        # Sequence information
        target['res'] = self.labels['res'][index][0]
        target['vid'] = self.labels['vid'][index]
        target['frame_id'] = self.labels['frame_id'][index][1:]
        
        # Camera information
        target['cam_intrinsics'] = self.compute_cam_intrinsics(target['res'])
        R = self.labels['cam_poses'][index][:, :3, :3].clone()
        if 'emdb' in self.data.lower():
            # Use groundtruth camera angular velocity.
            # Can be updated with SLAM results if you have it.
            cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
            cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)) * FPS
            target['R'] = R
        else:
            cam_angvel = torch.zeros((len(target['pose']) - 1, 6))
        target['cam_angvel'] = cam_angvel
        return target

    def prepare_inputs(self, index, target):
        for key in ['features', 'bbox']:
            data = self.labels[self.prefix + key][index][1:]
            target[key] = data
        
        bbox = self.labels[self.prefix + 'bbox'][index][..., [0, 1, -1]].clone().float()
        bbox[:, 2] = bbox[:, 2] / 200
        
        # Normalize keypoints
        kp2d, bbox = self.keypoints_normalizer(
            self.labels[self.prefix + 'kp2d'][index][..., :2].clone().float(), 
            target['res'], target['cam_intrinsics'], 224, 224, bbox)
        target['kp2d'] = kp2d
        target['bbox'] = bbox[1:]
        
        # Masking out low confident keypoints
        mask = self.labels[self.prefix + 'kp2d'][index][..., -1] < 0.3
        target['input_kp2d'] = self.labels['kp2d'][index][1:]
        target['input_kp2d'][mask[1:]] *= 0
        target['mask'] = mask[1:]
        
        return target

    def prepare_initialization(self, index, target):
        # Initial frame per-frame estimation
        target['init_kp3d'] = root_centering(self.labels[self.prefix + 'init_kp3d'][index][:1, :self.n_joints]).reshape(1, -1)
        target['init_pose'] = transforms.axis_angle_to_matrix(self.labels[self.prefix + 'init_pose'][index][:1]).cpu()
        pose_root = target['pose'][:, 0].clone()
        target['init_root'] = transforms.matrix_to_rotation_6d(pose_root)
        
        return target
        
    def get_data(self, index):
        target = {}
        
        target = self.prepare_labels(index, target)
        target = self.prepare_inputs(index, target)
        target = self.prepare_initialization(index, target)
        
        return target