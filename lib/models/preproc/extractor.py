from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import cv2
import torch
import numpy as np
import scipy.signal as signal
from progress.bar import Bar
from scipy.ndimage.filters import gaussian_filter1d

from configs import constants as _C
from .backbone.hmr2 import hmr2
from .backbone.utils import process_image

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")

class FeatureExtractor(object):
    def __init__(self, device, max_batch_size=64):
        
        self.device = device
        self.max_batch_size = max_batch_size
        
        ckpt = osp.join(ROOT_DIR, 'checkpoints', 'hmr2a.ckpt')
        self.model = hmr2(ckpt).to(device).eval()
        
    def run(self, video, tracking_results, patch_h=256, patch_w=256):
        
        cap = cv2.VideoCapture(video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        frame_id = 0
        bar = Bar('Feature extraction ...', fill='#', max=length)
        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag:
                break
            
            for _id, val in tracking_results.items():
                if not frame_id in val['frame_id']: continue
                
                frame_id2 = np.where(val['frame_id'] == frame_id)[0][0]
                bbox = val['bbox'][frame_id2]
                cx, cy, scale = bbox
                
                norm_img, crop_img = process_image(img[..., ::-1], [cx, cy], scale, patch_h, patch_w)
                norm_img = torch.from_numpy(norm_img).unsqueeze(0).to(self.device)
                feature = self.model(norm_img, encode=True)
                tracking_results[_id]['features'].append(feature.cpu())
                
                if frame_id2 == 0: # First frame of this subject
                    pred_global_orient, pred_body_pose, pred_betas, _ = self.model(norm_img, encode=False)
                    tracking_results[_id]['init_global_orient'] = pred_global_orient.cpu()
                    tracking_results[_id]['init_body_pose'] = pred_body_pose.cpu()
                    tracking_results[_id]['init_betas'] = pred_betas.cpu()
        
            bar.next()
            frame_id += 1
        
        return tracking_results