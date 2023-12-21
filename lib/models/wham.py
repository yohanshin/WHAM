from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn

from configs import constants as _C
from lib.utils import transforms
from lib.models.layers import (MotionEncoder, MotionDecoder, TrajectoryDecoder, TrajectoryRefiner, Integrator, 
                               rollout_global_motion, compute_camera_pose, reset_root_velocity)


class Network(nn.Module):
    def __init__(self, 
                 smpl,
                 pose_dr=0.1,
                 d_embed=512,
                 n_layers=3,
                 d_feat=2048,
                 rnn_type='LSTM',
                 **kwargs
                 ):
        super().__init__()
        
        n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.smpl = smpl
        in_dim = n_joints * 2 + 3
        d_context = d_embed + n_joints * 3
        
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, n_joints, 2))        
        
        # Module 1. Motion Encoder
        self.motion_encoder = MotionEncoder(in_dim=in_dim, 
                                            d_embed=d_embed,
                                            pose_dr=pose_dr,
                                            rnn_type=rnn_type,
                                            n_layers=n_layers,
                                            n_joints=n_joints)
        
        self.trajectory_decoder = TrajectoryDecoder(d_embed=d_context,
                                                    rnn_type=rnn_type,
                                                    n_layers=n_layers)
        
        # Module 3. Feature Integrator
        self.integrator = Integrator(in_channel=d_feat + d_context, 
                                     out_channel=d_context)
        
        # Module 4. Motion Decoder
        self.motion_decoder = MotionDecoder(d_embed=d_context,
                                            rnn_type=rnn_type,
                                            n_layers=n_layers)
        
        # Module 5. Trajectory Refiner
        self.trajectory_refiner = TrajectoryRefiner(d_embed=d_context,
                                                    d_hidden=d_embed,
                                                    rnn_type=rnn_type,
                                                    n_layers=2)
        
    @property
    def __version__(self, ):
        return 'v07'
    
    def forward_smpl(self, **kwargs):
        self.output = self.smpl(self.pred_pose, 
                                self.pred_shape,
                                cam=self.pred_cam,
                                return_full_pose=not self.training,
                                **kwargs,
                                )
        kp3d = self.output.joints
        
        # Global motion
        root_world, trans = rollout_global_motion(self.pred_root, self.pred_vel)
        
        # Compute world-coordinate motion
        self.global_output = self.smpl.get_output(
            global_orient=root_world.reshape(self.b * self.f, 1, 3, 3), body_pose=self.output.body_pose,
            betas=self.output.betas, pose2rot=False
        )
        feet_world = self.global_output.feet.reshape(self.b, self.f, 4, 3) + trans.unsqueeze(-2)
        
        output = {'feet': feet_world,
                  'contact': self.pred_contact,
                  'pose': self.pred_pose, 
                  'betas': self.pred_shape, 
                  'poses_root_cam': self.output.global_orient,
                  'verts': self.output.vertices}
        if self.training:
            pass     # TODO: Update training code
        else:
            pose = transforms.matrix_to_axis_angle(self.output.full_pose).reshape(-1, 72)
            theta = torch.cat((self.output.full_cam, pose, self.pred_shape.squeeze(0)), dim=-1)
            output.update({
                'poses_root_r6d': self.pred_root,
                'trans_cam': self.output.full_cam,
                'poses_body': self.output.body_pose})
        
        return output        
    
    def forward(self, x, inits, img_features=None, mask=None, init_root=None, cam_angvel=None,
                cam_intrinsics=None, bbox=None, res=None, **kwargs):
        self.b, self.f = x.shape[:2]
        init_kp, init_smpl = inits
        
        # Treat masked keypoints
        mask_embedding = mask.unsqueeze(-1) * self.mask_embedding
        _mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(self.b, self.f, -1)
        _mask = torch.cat((_mask, torch.zeros_like(_mask[..., :3])), dim=-1)
        _mask_embedding = mask_embedding.reshape(self.b, self.f, -1)
        _mask_embedding = torch.cat((_mask_embedding, torch.zeros_like(_mask_embedding[..., :3])), dim=-1)
        x[_mask] = 0.0
        x = x + _mask_embedding
        
        # --------- Inference --------- #
        # Stage 1. Encode motion
        pred_kp3d, motion_context = self.motion_encoder(x, init_kp)
        old_motion_context = motion_context.detach().clone()
        
        # Stage 2. Decode global trajectory
        pred_root, pred_vel = self.trajectory_decoder(motion_context, init_root, cam_angvel)
        
        # Stage 3. Integrate features
        if img_features is not None and self.integrator is not None:
            motion_context = self.integrator(motion_context, img_features)
            
        # Stage 4. Decode SMPL motion
        pred_pose, pred_shape, pred_cam, pred_contact = self.motion_decoder(motion_context, init_smpl)
        # --------- #
        
        # --------- Register predictions --------- #
        self.pred_kp3d = pred_kp3d
        self.pred_root = pred_root
        self.pred_vel = pred_vel
        self.pred_pose = pred_pose
        self.pred_shape = pred_shape
        self.pred_cam = pred_cam
        self.pred_contact = pred_contact
        # --------- #
        
        # --------- Build SMPL --------- #
        output = self.forward_smpl(cam_intrinsics=cam_intrinsics, bbox=bbox, res=res)
        # --------- #
        
        # --------- Refine trajectory --------- #
        update_vel = reset_root_velocity(self.smpl, self.output, pred_contact, pred_root, pred_vel, thr=0.5)
        output = self.trajectory_refiner(old_motion_context, update_vel, output, cam_angvel)
        # --------- #
        
        return output