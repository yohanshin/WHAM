from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib
from lib.utils import transforms

from configs import constants as _C

from .amass import compute_contact_label, perspective_projection
from ..utils.augmentor import *
from .._dataset import BaseDataset
from ...models import build_body_model
from ...utils import data_utils as d_utils
from ...utils.kp_utils import root_centering

class BEDLAMDataset(BaseDataset):
    def __init__(self, cfg):
        label_pth = _C.PATHS.BEDLAM_LABEL.replace('backbone', cfg.MODEL.BACKBONE)
        super(BEDLAMDataset, self).__init__(cfg, training=True)

        self.labels = joblib.load(label_pth)
        
        self.VideoAugmentor = VideoAugmentor(cfg)
        self.SMPLAugmentor = SMPLAugmentor(cfg, False)
        
        self.smpl = build_body_model('cpu', self.n_frames)
        self.prepare_video_batch()

    @property
    def __name__(self, ):
        return 'BEDLAM'

    def get_inputs(self, index, target, vis_thr=0.6):
        start_index, end_index = self.video_indices[index]
        
        bbox = self.labels['bbox'][start_index:end_index+1].clone()
        bbox[:, 2] = bbox[:, 2] / 200
        
        gt_kp3d = target['kp3d']
        inpt_kp3d = self.VideoAugmentor(gt_kp3d[:, :self.n_joints, :-1].clone())
        # kp2d = perspective_projection(inpt_kp3d, target['K'])
        kp2d = perspective_projection(inpt_kp3d, self.cam_intrinsics)
        mask = self.VideoAugmentor.get_mask()
        # kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], self.cam_intrinsics, 224, 224, bbox)
        kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], self.cam_intrinsics, 224, 224)

        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d
        target['mask'] = mask[1:]
        
        # Image features
        target['features'] = self.labels['features'][start_index+1:end_index+1].clone()
        
        return target

    def get_groundtruth(self, index, target):
        start_index, end_index = self.video_indices[index]

        # GT 1. Joints
        gt_kp3d = target['kp3d']
        # gt_kp2d = perspective_projection(gt_kp3d, target['K'])
        gt_kp2d = perspective_projection(gt_kp3d, self.cam_intrinsics)
        target['kp3d'] = torch.cat((gt_kp3d, torch.ones_like(gt_kp3d[..., :1])), dim=-1)
        # target['full_kp2d'] = torch.cat((gt_kp2d, torch.zeros_like(gt_kp2d[..., :1])), dim=-1)[1:]
        target['full_kp2d'] = torch.cat((gt_kp2d, torch.ones_like(gt_kp2d[..., :1])), dim=-1)[1:]
        target['weak_kp2d'] = torch.zeros_like(target['full_kp2d'])
        target['init_kp3d'] = root_centering(gt_kp3d[:1, :self.n_joints].clone()).reshape(1, -1)
        
        # GT 2. Root pose
        w_transl = self.labels['w_trans'][start_index:end_index+1]
        pose_root = transforms.axis_angle_to_matrix(self.labels['root'][start_index:end_index+1])
        vel_world = (w_transl[1:] - w_transl[:-1])
        vel_root = (pose_root[:-1].transpose(-1, -2) @ vel_world.unsqueeze(-1)).squeeze(-1)
        target['vel_root'] = vel_root.clone()
        target['pose_root'] = transforms.matrix_to_rotation_6d(pose_root)
        target['init_root'] = target['pose_root'][:1].clone()

        return target

    def forward_smpl(self, target):
        output = self.smpl.get_output(
            body_pose=torch.cat((target['init_pose'][:, 1:], target['pose'][1:, 1:])),
            global_orient=torch.cat((target['init_pose'][:, :1], target['pose'][1:, :1])),
            betas=target['betas'],
            transl=target['transl'],
            pose2rot=False)
        
        target['kp3d'] = output.joints + output.offset.unsqueeze(1)
        target['feet'] = output.feet[1:] + target['transl'][1:].unsqueeze(-2)
        target['verts'] = output.vertices[1:, ].clone()
        
        return target

    def augment_data(self, target):
        # Augmentation 1. SMPL params augmentation
        target = self.SMPLAugmentor(target)
        
        # Get world-coordinate SMPL
        target = self.forward_smpl(target)
        
        return target

    def load_camera(self, index, target):
        start_index, end_index = self.video_indices[index]

        # Get camera info
        extrinsics = self.labels['extrinsics'][start_index:end_index+1].clone()
        R = extrinsics[:, :3, :3]
        T = extrinsics[:, :3, -1]
        K = self.labels['intrinsics'][start_index:end_index+1].clone()
        width, height = K[0, 0, 2] * 2, K[0, 1, 2] * 2
        target['R'] = R
        target['res'] = torch.tensor([width, height]).float()
        
        # Compute angular velocity
        cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
        cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel) # Normalize
        target['cam_angvel'] = cam_angvel * 3e1 # BEDLAM is 30-fps
        
        target['K'] = K # Use GT camera intrinsics for projecting keypoints 
        self.get_naive_intrinsics(target['res'])
        target['cam_intrinsics'] = self.cam_intrinsics

        return target

    def load_params(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # Load AMASS labels
        pose = self.labels['pose'][start_index:end_index+1].clone()
        pose = transforms.axis_angle_to_matrix(pose.reshape(-1, 24, 3))
        transl = self.labels['c_trans'][start_index:end_index+1].clone()
        betas = self.labels['betas'][start_index:end_index+1, :10].clone()
        
        # Stack GT
        target.update({'vid': self.labels['vid'][start_index].clone(), 
                       'pose': pose, 
                       'transl': transl, 
                       'betas': betas})
        
        return target


    def get_single_sequence(self, index):
        target = {'has_full_screen': torch.tensor(True),
                  'has_smpl': torch.tensor(True),
                  'has_traj': torch.tensor(False),
                  'has_verts': torch.tensor(True),
                  
                  # Null contact label
                  'contact': torch.ones((self.n_frames - 1, 4)) * (-1),
                  }
        
        target = self.load_params(index, target)
        target = self.load_camera(index, target)
        target = self.augment_data(target)
        target = self.get_groundtruth(index, target)
        target = self.get_inputs(index, target)
        
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target