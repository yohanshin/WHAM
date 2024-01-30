# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import yaml
import torch
import shutil
from os import path as osp


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logdir = osp.join(cfg.OUTPUT_DIR, cfg.EXP_NAME)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg


def prepare_groundtruth(batch, device):
    groundtruths = dict()
    gt_keys = ['pose', 'cam', 'betas', 'kp3d', 'mask', 'bbox', 'res', 'cam_intrinsics', 'init_root', 'cam_angvel']
    for gt_key in gt_keys:
        if gt_key in batch.keys():
            dtype = torch.float32 if batch[gt_key].dtype == torch.float64 else batch[gt_key].dtype
            groundtruths[gt_key] = batch[gt_key].to(dtype=dtype, device=device)
    
    return groundtruths


def prepare_input(batch, device, use_features):
    # Input keypoints data
    kp2d = batch['kp2d'].to(device).float()

    # Input features
    if use_features and 'features' in batch.keys():
        features = batch['features'].to(device).float()
    else:
        features = None

    # Initial SMPL parameters
    init_smpl = batch['init_pose'].to(device).float()

    # Initial keypoints
    init_kp = torch.cat((
        batch['init_kp3d'], batch['init_kp2d']
    ), dim=-1).to(device).float()

    return kp2d, (init_kp, init_smpl), features


def prepare_batch(batch, device, use_features=True):
    x, inits, features = prepare_input(batch, device, use_features)
    groundtruths = prepare_groundtruth(batch, device)
    
    return x, inits, features, groundtruths