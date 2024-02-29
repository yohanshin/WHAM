from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib
import numpy as np

from .._dataset import BaseDataset
from ..utils.augmentor import *
from ...utils import data_utils as d_utils
from ...utils import transforms
from ...models import build_body_model
from ...utils.kp_utils import convert_kps, root_centering


class Dataset3D(BaseDataset):
    def __init__(self, cfg, fname, training):
        super(Dataset3D, self).__init__(cfg, training)

        self.epoch = 0
        self.labels = joblib.load(fname)
        self.n_frames = cfg.DATASET.SEQLEN + 1

        if self.training:
            self.prepare_video_batch()

        self.smpl = build_body_model('cpu', self.n_frames)
        self.SMPLAugmentor = SMPLAugmentor(cfg, False)
        self.VideoAugmentor = VideoAugmentor(cfg)

    def __getitem__(self, index):
        return self.get_single_sequence(index)

    def load_train_data(self, index):
        target = {}
        s, e = self.video_indices[index]
        
        # ========== Base labels ==========
        target['res'] = self.labels['res'][s:e+1][0].clone()
        target['cam_intrinsics'] = self.compute_cam_intrinsics(target['res'])

        target['pose'] = transforms.axis_angle_to_matrix(
            self.labels['pose'][s:e+1].clone().reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][s:e+1].clone()        # No t
        target = self.SMPLAugmentor(target)
        
        kp2d = self.labels['kp2d'][s:e+1][..., :2].clone()
        bbox = self.labels['bbox'][s:e+1][..., [0, 1, -1]].clone()
        bbox[:, 2] = bbox[:, 2] / 200
        
        if self.labels['kp2d'].shape[-1] == 3:
            mask = self.labels['kp2d'][s+1:e+1][..., -1] < 0.6
        else:
            mask = torch.zeros((self.n_frames - 1, 17)).bool()
        
        # mask = None
        target['input_kp2d'] = kp2d[1:].clone()
        kp2d, bbox = self.keypoints_normalizer( 
            kp2d, target['res'], target['cam_intrinsics'], 224, 224, bbox)
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d

        if self.__name__ == 'ThreeDPW':
            gt_kp3d = self.labels['joints3D'][s:e+1].clone()
            gt_kp2d = self.labels['joints2D'][s+1:e+1, ..., :2].clone()
            gt_kp3d = root_centering(gt_kp3d.clone())
            
        else: 
            if self.has_smpl:
                gt_kp3d = root_centering(self.labels['joints3D_smpl'][s:e+1].clone().float())
                gt_kp2d = self.labels['gt_kp2d'][s+1:e+1].clone().float()
            else:
                gt_kp3d = torch.zeros((self.n_frames, self.n_joints + 14, 3))
                gt_kp3d[:, self.n_joints:] = convert_kps(self.labels['joints3D'][s:e+1], 'spin', 'common')
                gt_kp2d = torch.zeros((self.n_frames - 1, self.n_joints + 14, 2))
                gt_kp2d[:, self.n_joints:] = convert_kps(self.labels['joints2D'][s+1:e+1, ..., :2], 'spin', 'common')

        conf = self.mask.repeat(self.n_frames, 1).unsqueeze(-1)        
        gt_kp2d = torch.cat((gt_kp2d, conf[1:]), dim=-1)
        gt_kp3d = torch.cat((gt_kp3d, conf), dim=-1)
        
        target['kp3d'] = gt_kp3d
        target['gt_full_kp2d'] = gt_kp2d
        target['gt_weak_kp2d'] = torch.zeros_like(gt_kp2d)
        
        if self.__name__ != 'ThreeDPW':
            target['stationary'] = self.labels['stationaries'][s+1:e+1].clone()
        else:
            target['stationary'] = torch.ones((self.n_frames - 1, 4)) * (-1)
        
        # Prepare initial frame
        output = self.smpl.get_output(
            body_pose=target['init_pose'][:, 1:],
            global_orient=target['init_pose'][:, :1],
            betas=target['betas'][:1],
            pose2rot=False
        )
        target['init_kp3d'] = root_centering(output.joints[:1, :self.n_joints]).reshape(1, -1)       

        ## Inputs
        target['features'] = self.labels['features'][s+1:e+1].clone()
        target['transl'] = torch.zeros((len(target['pose']), 3))
        target['has_smpl'] = torch.tensor(self.has_smpl)
        target['has_full_screen'] = torch.tensor(True)
        target['mask'] = mask
        
        ###### 07. 27. 2023 Add camera angular velocity as additional condition
        R = self.labels['cam_poses'][s:e+1, :3, :3].clone().float()
        yaw = transforms.axis_angle_to_matrix(torch.tensor([[0, 2 * np.pi * np.random.uniform(), 0]])).float()
        if self.__name__ == 'Human36M':
            # Map Z-up to Y-down coordinate
            zup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[-np.pi/2, 0, 0]])).float()
            zup2ydown = torch.matmul(yaw, zup2ydown)
            R = torch.matmul(R, zup2ydown)
        elif self.__name__ == 'MPII3D':
            # Map Y-up to Y-down coordinate
            yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float()
            yup2ydown = torch.matmul(yaw, yup2ydown)
            R = torch.matmul(R, yup2ydown)
        
        # R = R @ R[:1].transpose(-1, -2) # For make it match with z-up axis        
        cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
        cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)
        target['cam_a'] = cam_angvel * 3e1
        target['cam_r'] = transforms.matrix_to_rotation_6d(R)
        
        ###### root orientation in world coordinate
        pose_root = target['pose'][:, 0].clone()
        pose_root = R.transpose(-1, -2) @ pose_root
        target['root_r'] = transforms.matrix_to_rotation_6d(pose_root)

        ###### root translation in world coordinate
        if self.__name__ == 'ThreeDPW':
            vel_root = torch.zeros((self.n_frames - 1, 3)).float()
            target['root_r'] = target['root_r'] * 0.0
            target['cam_r'] = target['cam_r'] * 0.0
        else:
            transl_cam = (self.labels['trans'][s:e+1].clone()).squeeze(1)
            vel_cam = transl_cam[1:] - transl_cam[:-1]
            vel_root = (target['pose'][:-1, 0].transpose(-1, -2) @ vel_cam.unsqueeze(-1)).squeeze(-1)
        target['root_v'] = vel_root.clone()
        target['dataset'] = self.__name__

        return target

    def get_single_sequence(self, index):
        target = {}

        if self.training:
            target = self.load_train_data(index)
        else:
            target = self.load_test_data(index)

        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target