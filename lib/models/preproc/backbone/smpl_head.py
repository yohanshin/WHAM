import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from configs import constants as _C
from lib.utils.transforms import axis_angle_to_matrix
from .pose_transformer import TransformerDecoder

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def build_smpl_head(cfg):
    smpl_head_type = 'transformer_decoder'
    if  smpl_head_type == 'transformer_decoder':
        return SMPLTransformerDecoderHead(cfg)
    else:
        raise ValueError('Unknown SMPL head type: {}'.format(smpl_head_type))

class SMPLTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    """

    def __init__(self):
        super().__init__()
        self.joint_rep_type = '6d'
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * 24
        self.npose = npose
        self.input_is_mean_shape = False
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args_from_cfg = dict(
            depth=6, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0, emb_dropout=0.0, norm='layer', context_dim=1280
        )
        transformer_args = (transformer_args | transformer_args_from_cfg)
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load(_C.BMODEL.MEAN_PARAMS)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        
        # Input token to transformer is zero token
        if len(x.shape) > 2:
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:,None,:]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1) # (B, C)
        else:
            token_out = x

        # Readout from token_out
        pred_body_pose = self.decpose(token_out) + pred_body_pose
        pred_betas = self.decshape(token_out) + pred_betas
        pred_cam = self.deccam(token_out) + pred_cam
        pred_body_pose_list.append(pred_body_pose)
        pred_betas_list.append(pred_betas)
        pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: axis_angle_to_matrix(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smpl_params_list = {}
        pred_smpl_params_list['body_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0)
        pred_smpl_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, 24, 3, 3)

        pred_smpl_params = {'global_orient': pred_body_pose[:, [0]],
                            'body_pose': pred_body_pose[:, 1:],
                            'betas': pred_betas}
        return pred_smpl_params, pred_cam, pred_smpl_params_list