from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib

from .._dataset import BaseDataset
from ..utils.augmentor import *
from ...utils import data_utils as d_utils
from ...utils import transforms
from ...models import build_body_model
from ...utils.kp_utils import convert_kps, root_centering


class Dataset2D(BaseDataset):
    def __init__(self, cfg, fname, training):
        super(Dataset2D, self).__init__(cfg, training)

        self.epoch = 0
        self.n_frames = cfg.DATASET.SEQLEN + 1
        self.labels = joblib.load(fname)

        if self.training:
            self.prepare_video_batch()
        
        self.smpl = build_body_model('cpu', self.n_frames)
        self.SMPLAugmentor = SMPLAugmentor(cfg, False)

    def __getitem__(self, index):
        return self.get_single_sequence(index)

    def get_inputs(self, index, target, vis_thr=0.6):
        start_index, end_index = self.video_indices[index]
        
        # 2D keypoints detection
        kp2d = self.labels['kp2d'][start_index:end_index+1][..., :2].clone()
        kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], target['cam_intrinsics'], 224, 224, target['bbox'])
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d
        
        # Detection mask
        target['mask'] = ~self.labels['joints2D'][start_index+1:end_index+1][..., -1].clone().bool()
        
        # Image features
        target['features'] = self.labels['features'][start_index+1:end_index+1].clone()
        
        return target
    
    def get_labels(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # SMPL parameters
        # NOTE: We use NeuralAnnot labels for Human36m and MPII3D only for the 0th frame input.
        #       We do not supervise the network on SMPL parameters.
        target['pose'] = transforms.axis_angle_to_matrix(
            self.labels['pose'][start_index:end_index+1].clone().reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][start_index:end_index+1].clone()        # No t
        
        # Apply SMPL augmentor (y-axis rotation and initial frame noise)
        target = self.SMPLAugmentor(target)
        
        # 2D keypoints
        kp2d = self.labels['kp2d'][start_index:end_index+1].clone().float()[..., :2]
        gt_kp2d = torch.zeros((self.n_frames - 1, 31, 2))
        gt_kp2d[:, :17] = kp2d[1:].clone()
        
        # Set 0 confidence to the masked keypoints
        mask = torch.zeros((self.n_frames - 1, 31))
        mask[:, :17] = self.labels['joints2D'][start_index+1:end_index+1][..., -1].clone()
        mask = torch.logical_and(gt_kp2d.mean(-1) != 0, mask)
        gt_kp2d = torch.cat((gt_kp2d, mask.float().unsqueeze(-1)), dim=-1)
        
        _gt_kp2d = gt_kp2d.clone()
        for idx in range(len(_gt_kp2d)):
            _gt_kp2d[idx][..., :2] = torch.from_numpy(
                self.j2d_processing(gt_kp2d[idx][..., :2].numpy().copy(),
                                    target['bbox'][idx].numpy().copy()))

        target['weak_kp2d'] = _gt_kp2d.clone()
        target['full_kp2d'] = torch.zeros_like(gt_kp2d)
        target['kp3d'] = torch.zeros((kp2d.shape[0], 31, 4))
        
        return target
        
    def get_init_frame(self, target):
        # Prepare initial frame
        output = self.smpl.get_output(
            body_pose=target['init_pose'][:, 1:],
            global_orient=target['init_pose'][:, :1],
            betas=target['betas'][:1],
            pose2rot=False
        )
        target['init_kp3d'] = root_centering(output.joints[:1, :self.n_joints]).reshape(1, -1)   
        
        return target

    def get_single_sequence(self, index):
        # Camera parameters
        res = (224.0, 224.0)
        bbox = torch.tensor([112.0, 112.0, 1.12])
        res = torch.tensor(res)
        self.get_naive_intrinsics(res)
        bbox = bbox.repeat(self.n_frames, 1)
        
        # Universal target
        target = {'has_smpl': torch.tensor(self.has_smpl),
                  'has_full_screen': torch.tensor(False),
                  'has_verts': torch.tensor(False),
                  'transl': torch.zeros((self.n_frames, 3)),
                  
                  # Camera parameters and bbox
                  'res': res,
                  'cam_intrinsics': self.cam_intrinsics,
                  'bbox': bbox,
                  
                  # Null camera motion
                  'R': torch.eye(3).repeat(self.n_frames, 1, 1),
                  'cam_angvel': torch.zeros((self.n_frames - 1, 6)),
                  
                  # Null root orientation and velocity
                  'pose_root': torch.zeros((self.n_frames, 6)),
                  'vel_root': torch.zeros((self.n_frames - 1, 3)),
                  'init_root': torch.zeros((1, 6)),
                  
                  # Null contact label
                  'contact': torch.ones((self.n_frames - 1, 4)) * (-1)
                  }
        
        self.get_inputs(index, target)
        self.get_labels(index, target)
        self.get_init_frame(target)
        
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)
        
        return target