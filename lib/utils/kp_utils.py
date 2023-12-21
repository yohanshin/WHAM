from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C

import torch


def root_centering(X, joint_type='coco'):
    """Center the root joint to the pelvis."""
    if joint_type != 'common' and X.shape[-2] == 14: return X
    
    conf = None
    if X.shape[-1] == 4:
        conf = X[..., -1:]
        X = X[..., :-1]
        
    if X.shape[-2] == 31:
        X[..., :17, :] = X[..., :17, :] - X[..., [12, 11], :].mean(-2, keepdims=True)
        X[..., 17:, :]  = X[..., 17:, :] - X[..., [19, 20], :].mean(-2, keepdims=True)
        
    elif joint_type == 'coco':
        X = X - X[..., [12, 11], :].mean(-2, keepdims=True)
    
    elif joint_type == 'common':
        X = X - X[..., [2, 3], :].mean(-2, keepdims=True)

    if conf is not None:
        X = torch.cat((X, conf), dim=-1)
    
    return X