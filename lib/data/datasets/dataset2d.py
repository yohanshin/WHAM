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

    def load_train_data(self, index):
        target = {}
        s, e = self.video_indices[index]

        ## Groundtruth
        kp2d = self.labels['kp2d'][s:e+1].clone().float()[..., :2]
        
        target['input_kp2d'] = kp2d[1:].clone()
        gt_kp2d = torch.zeros((self.n_frames - 1, 31, 2))
        gt_kp2d[:, :17] = kp2d[1:].clone()
        
        res = (224.0, 224.0)
        bbox = torch.tensor([112.0, 112.0, 1.12])
        target['res'] = torch.tensor(res)
        target['cam_intrinsics'] = self.compute_cam_intrinsics(target['res'])
        target['bbox'] = bbox.repeat(self.n_frames, 1)
        kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], target['cam_intrinsics'], 224, 224, target['bbox'])
        
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d
                        
        target['features'] = self.labels['features'][s+1:e+1]
        mask = torch.zeros((self.n_frames - 1, 31))
        mask[:, :17] = self.labels['joints2D'][s+1:e+1][..., -1].clone()
        mask = torch.logical_and(gt_kp2d.mean(-1) != 0, mask)
        gt_kp2d = torch.cat((gt_kp2d, mask.float().unsqueeze(-1)), dim=-1)
        target['gt_full_kp2d'] = torch.zeros_like(gt_kp2d)
        
        _gt_kp2d = gt_kp2d.clone()
        for idx in range(len(_gt_kp2d)):
            _gt_kp2d[idx][..., :2] = torch.from_numpy(
                self.j2d_processing(gt_kp2d[idx][..., :2].numpy().copy(),
                                    target['bbox'][idx].numpy().copy()))
        target['gt_weak_kp2d'] = _gt_kp2d.clone()

        target['kp3d'] = torch.zeros((kp2d.shape[0], 31, 4))
        target['pose'] = transforms.axis_angle_to_matrix(
            self.labels['pose'][s:e+1].clone().reshape(-1, 24, 3)
        )
        target['betas'] = self.labels['betas'][s:e+1]        # No t
        target = self.SMPLAugmentor(target)

        # Get 3D init params
        output = self.smpl.get_output(
            body_pose=target['init_pose'][:, 1:],
            global_orient=target['init_pose'][:, :1],
            betas=target['betas'][:1],
            pose2rot=False
        )
        target['init_kp3d'] = root_centering(output.joints[:, :17]).reshape(1, -1)
        target['transl'] = torch.zeros((len(target['pose']), 3))
        target['has_smpl'] = torch.tensor(False)
        target['has_full_screen'] = torch.tensor(False)

        # 2D dataset does not have stationary info
        target['stationary'] = torch.ones((self.n_frames - 1, 4)) * (-1)
        target['mask'] = ~self.labels['joints2D'][s+1:e+1][..., -1].clone().bool()
        
        ###### 07. 27. 2023 Add camera angular velocity as additional condition
        target['cam_r'] = torch.zeros((self.n_frames, 6)).float()
        target['cam_a'] = torch.zeros((self.n_frames - 1, 6)).float()
        target['root_v'] = torch.zeros((self.n_frames - 1, 3)).float()
        target['root_r'] = torch.zeros((self.n_frames, 6)).float()
        target['campose'] = torch.zeros(2, ).float()
        target['cam_move'] = torch.tensor(True)
        target['has_verts'] = torch.tensor(False)
        target['verts'] = torch.zeros((len(target['features']), 6890, 3)).float()

        target['dataset'] = torch.tensor(['AMASS', 'ThreeDPW', 'Human36M', 'MPII3D', 'InstaVariety'].index(self.__name__))
        return target

    def get_single_sequence(self, index):
        target = {}
        target = self.load_train_data(index)

        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target
