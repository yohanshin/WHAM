from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np

from lib.utils import transforms


def make_collate_fn():
    def collate_fn(items):
        items = list(filter(lambda x: x is not None , items))
        batch = dict()
        try: batch['vid'] = [item['vid'] for item in items]
        except: pass
        try: batch['gender'] = [item['gender'] for item in items]
        except: pass
        for key in items[0].keys():
            try: batch[key] = torch.stack([item[key] for item in items])
            except: pass
        return batch

    return collate_fn


def prepare_keypoints_data(target):
    """Prepare keypoints data"""
    
    # Prepare 2D keypoints
    target['init_kp2d'] = target['kp2d'][:1]
    target['kp2d'] = target['kp2d'][1:]
    if 'kp3d' in target:
        target['kp3d'] = target['kp3d'][1:]

    return target


def prepare_smpl_data(target):
    if 'pose' in target.keys():
        # Use only the main joints
        pose = target['pose'][:]
        # 6-D Rotation representation
        pose6d = transforms.matrix_to_rotation_6d(pose)
        target['pose'] = pose6d[1:]
    
    if 'betas' in target.keys():
        target['betas'] = target['betas'][1:]
    
    # Translation and shape parameters
    if 'transl' in target.keys():
        target['cam'] = target['transl'][1:]
    
    # Initial pose and translation
    target['init_pose'] = transforms.matrix_to_rotation_6d(target['init_pose'])

    return target


def append_target(target, label, key_list, idx1, idx2=None, pad=True):
    for key in key_list:
        if idx2 is None: data = label[key][idx1]
        else: data = label[key][idx1:idx2+1]
        if not pad: data = data[2:]
        target[key] = data
        
    return target


def map_dmpl_to_smpl(pose):
    """ Map AMASS DMPL pose representation to SMPL pose representation

    Args:
        pose - tensor / array with shape of (n_frames, 156)

    Return:
        pose - tensor / array with shape of (n_frames, 24, 3)
    """

    pose = pose.reshape(pose.shape[0], -1, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    if isinstance(pose, np.ndarray): pose = pose[:, :24].copy()
    else: pose = pose[:, :24].clone()
    return pose


def transform_global_coordinate(pose, T, transl=None):
    """ Transform global coordinate of dataset with respect to the given matrix.
    Various datasets have different global coordinate system, 
    thus we united all datasets to the cronical coordinate system.

    Args:
        pose - SMPL pose; tensor / array
        T - Transformation matrix
        transl - SMPL translation
    """

    return_to_numpy = False
    if isinstance(pose, np.ndarray):
        return_to_numpy = True
        pose = torch.from_numpy(pose).float()
        if transl is not None: transl = torch.from_numpy(transl).float()

    pose = transforms.axis_angle_to_matrix(pose)
    pose[:, 0] = T @ pose[:, 0]
    pose = transforms.matrix_to_axis_angle(pose)
    if transl is not None:
        transl = (T @ transl.T).squeeze().T

    if return_to_numpy:
        pose = pose.detach().numpy()
        if transl is not None: transl = transl.detach().numpy()
    return pose, transl