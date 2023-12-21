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
from lib.models import build_body_model
from lib.utils.kp_utils import root_centering
from lib.data._dataset import BaseDataset

FPS = 30
class EvalDataset(BaseDataset):
    def __init__(self, cfg, data, split, backbone):
        super(EvalDataset, self).__init__(cfg, False)
        
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'{data}_{split}_{backbone}.pth')
        self.labels = joblib.load(parsed_data_path)

    def __getitem__(self, index):
        target = {}
        target = self.load_data(index)
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target

    def __len__(self):
        return len(self.labels['kp2d'])

    def load_data(self, index):
        target = {}
        
        target['res'] = self.labels['res'][index][0]
        for key in ['features', 'bbox']:
            data = self.labels[key][index][1:]
            target[key] = data
        
        target['pose'] = transforms.axis_angle_to_matrix(self.labels['pose'][index].reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][index]
        target['gender'] = self.labels['gender'][index]
        target['cam_intrinsics'] = self.compute_cam_intrinsics(target['res'])

        bbox = self.labels['bbox'][index][..., [0, 1, -1]].clone().float()
        bbox[:, 2] = bbox[:, 2] / 200

        target['input_kp2d'] = self.labels['kp2d'][index][..., :-1].clone()[1:]
        kp2d, bbox = self.keypoints_normalizer(
            self.labels['kp2d'][index][..., :2].clone().float(), 
            target['res'], target['cam_intrinsics'], 224, 224, bbox)

        target['kp2d'] = kp2d
        target['bbox'] = bbox[1:]
        
        # Initial frame per-frame estimation
        target['init_kp3d'] = root_centering(self.labels['init_kp3d'][index][:1, :self.n_joints]).reshape(1, -1)
        target['init_pose'] = transforms.axis_angle_to_matrix(self.labels['init_pose'][index][:1]).cpu()

        # Compute ground truth 3D keypoints
        target['kp3d'] = self.labels['joints3D'][index]

        # Seqname
        target['vid'] = self.labels['vid'][index]
        target['frame_id'] = self.labels['frame_id'][index][1:]
        target['valid'] = torch.ones((len(target['bbox']), 1)).bool()
        
        mask = self.labels['kp2d'][index][..., -1] < 0.3
        target['input_kp2d'][mask[1:]] *= 0
        target['mask'] = mask[1:]
        
        R = self.labels['cam_poses'][index][:, :3, :3].clone()
        pose_root = target['pose'][:, 0].clone()
        target['init_root'] = transforms.matrix_to_rotation_6d(pose_root)

        if 'cam_angvel' in self.labels.keys():
            cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
            cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)) * FPS
        else:
            cam_angvel = torch.zeros((len(target['pose']) - 1, 6))
        target['cam_angvel'] = cam_angvel

        return target