import os

import torch
import einops
import torch.nn as nn
# import pytorch_lightning as pl

from yacs.config import CfgNode
from .vit import vit
from .smpl_head import SMPLTransformerDecoderHead

# class HMR2(pl.LightningModule):
class HMR2(nn.Module):

    def __init__(self):
        """
        Setup HMR2 model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Create backbone feature extractor
        self.backbone = vit()

        # Create SMPL head
        self.smpl_head = SMPLTransformerDecoderHead()


    def decode(self, x):
        
        batch_size = x.shape[0]
        pred_smpl_params, pred_cam, _ = self.smpl_head(x)

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        return pred_smpl_params['global_orient'], pred_smpl_params['body_pose'], pred_smpl_params['betas'], pred_cam

    def forward(self, x, encode=False, **kwargs):
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:,:,:,32:-32])
        if encode:
            conditioning_feats = einops.rearrange(conditioning_feats, 'b c h w -> b (h w) c')
            token = torch.zeros(batch_size, 1, 1).to(x.device)
            token_out = self.smpl_head.transformer(token, context=conditioning_feats)
            return token_out.squeeze(1)

        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_feats)

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        return pred_smpl_params['global_orient'], pred_smpl_params['body_pose'], pred_smpl_params['betas'], pred_cam
    
    
def hmr2(checkpoint_pth):
    model = HMR2()
    if os.path.exists(checkpoint_pth):
        model.load_state_dict(torch.load(checkpoint_pth, map_location='cpu')['state_dict'], strict=False)
        print(f'Load backbone weight: {checkpoint_pth}')
    return model