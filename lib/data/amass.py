from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import torch
import joblib
import numpy as np

from configs import constants as _C
from lib.utils import transforms

def foot_stationary_label(feet, joints, thr=1e-2, alpha=5):
    vel = torch.zeros_like(feet[..., 0])
    label = torch.zeros_like(feet[..., 0])
    
    vel1 = torch.norm(feet[1:-1] - feet[:-2], dim=-1)
    vel2 = torch.norm(feet[2:] - feet[1:-1], dim=-1)
    vel[1:-1] = (feet[2:] - feet[:-2]).norm(dim=-1) / 2.0
    vel[0] = vel[1].clone()
    vel[-1] = vel[-2].clone()
    
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    
    left_leg = joints[:, 15] - joints[:, 13]
    right_leg = joints[:, 16] - joints[:, 14]
    left_mask = left_leg[..., 1] / left_leg.norm(dim=-1) > 0.5 ** 0.5
    right_mask = right_leg[..., 1] / right_leg.norm(dim=-1) > 0.5 ** 0.5
    
    label[:, :2] = label[:, :2] * left_mask[1:].unsqueeze(-1)
    label[:, 2:] = label[:, 2:] * right_mask[1:].unsqueeze(-1)
    return label