from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

IMG_FEAT_DIM = {
    'resnet': 2048,
    'vit': 1024
}

N_JOINTS = 17
root = 'dataset'
class PATHS:
    PARSED_DATA = f'{root}/parsed_data'
    THREEDPW_PTH = f'{root}/3DPW'
    RICH_PTH = f'{root}/RICH'
    EMDB_PTH = f'{root}/EMDB'

class KEYPOINTS:
    NUM_JOINTS = N_JOINTS
    H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
    H36M_TO_J14 = H36M_TO_J17[:14]
    J17_TO_H36M = [14, 3, 4, 5, 2, 1, 0, 15, 12, 16, 13, 9, 10, 11, 8, 7, 6]
    COCO_AUG_DICT = f'{root}/body_models/coco_aug_dict.pth'
    TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]

    # STD scale for video noise
    S_BIAS = 1e-1
    S_JITTERING = 5e-2
    S_PEAK = 3e-1
    S_PEAK_MASK = 5e-3
    S_MASK = 0.03


class BMODEL:
    MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]    # reduced_joints

    FLDR = f'{root}/body_models/smpl/'
    SMPLX2SMPL = f'{root}/body_models/smplx2smpl.pkl'
    FACES = f'{root}/body_models/smpl_faces.npy'
    MEAN_PARAMS = f'{root}/body_models/smpl_mean_params.npz'
    JOINTS_REGRESSOR_WHAM = f'{root}/body_models/J_regressor_wham.npy'
    JOINTS_REGRESSOR_H36M = f'{root}/body_models/J_regressor_h36m.npy'
    JOINTS_REGRESSOR_EXTRA = f'{root}/body_models/J_regressor_extra.npy'
    JOINTS_REGRESSOR_FEET = f'{root}/body_models/J_regressor_feet.npy'
    PARENTS = torch.tensor([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])