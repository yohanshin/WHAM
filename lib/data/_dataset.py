from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from skimage.util.shape import view_as_windows

from configs import constants as _C
from .normalizer import Normalizer
from lib.utils.imutils import transform

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, training=True):
        super(BaseDataset, self).__init__()
        self.n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.epoch = 0
        self.n_frames = cfg.DATASET.SEQLEN + 1
        self.training = training
        self.keypoints_normalizer = Normalizer(cfg)

    def prepare_video_batch(self):
        r = self.epoch % 4

        self.video_indices = []
        vid_name = self.labels['vid']
        if isinstance(vid_name, torch.Tensor): vid_name = vid_name.numpy()
        video_names_unique, group = np.unique(
            vid_name, return_index=True)
        perm = np.argsort(group)
        group_perm = group[perm]
        indices = np.split(
            np.arange(0, self.labels['vid'].shape[0]), group_perm[1:]
        )
        for idx in range(len(video_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.n_frames: continue
            chunks = view_as_windows(
                indexes, (self.n_frames), step=self.n_frames // 4
            )
            start_finish = chunks[r::4, (0, -1)].tolist()
            self.video_indices += start_finish

        self.epoch += 1

    def __len__(self):
        if self.training:
            return len(self.video_indices)
        else:
            return len(self.labels['kp2d'])

    def __getitem__(self, index):
        return self.get_single_sequence(index)

    def get_single_sequence(self, index):
        NotImplementedError('get_single_sequence is not implemented')
        
    def compute_cam_intrinsics(self, res):
        img_w, img_h = res
        focal_length = (img_w * img_w + img_h * img_h) ** 0.5
        cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
        cam_intrinsics[:, 0, 0] = focal_length
        cam_intrinsics[:, 1, 1] = focal_length
        cam_intrinsics[:, 0, 2] = img_w/2.
        cam_intrinsics[:, 1, 2] = img_h/2.
        return cam_intrinsics
    
    def j2d_processing(self, kp, bbox):
        center = bbox[..., :2]
        scale = bbox[..., -1:]
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [224, 224])
        kp[:, :2] = 2. * kp[:, :2] / 224 - 1.
        kp = kp.astype('float32')
        return kp