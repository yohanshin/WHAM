from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from torch import nn
from configs import constants as _C
from .utils import rollout_global_motion
from lib.utils.transforms import axis_angle_to_matrix


class Regressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims, init_dim, layer='LSTM', n_layers=2, n_iters=1):
        super().__init__()
        self.n_outs = len(out_dims)

        self.rnn = getattr(nn, layer.upper())(
            in_dim + init_dim, hid_dim, n_layers, 
            bidirectional=False, batch_first=True, dropout=0.3)

        for i, out_dim in enumerate(out_dims):
            setattr(self, 'declayer%d'%i, nn.Linear(hid_dim, out_dim))
            nn.init.xavier_uniform_(getattr(self, 'declayer%d'%i).weight, gain=0.01)

    def forward(self, x, inits, h0):
        xc = torch.cat([x, *inits], dim=-1)
        xc, h0 = self.rnn(xc, h0)

        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d'%j)(xc)
            preds.append(out)

        return preds, xc, h0
    
    
class NeuralInitialization(nn.Module):
    def __init__(self, in_dim, hid_dim, layer, n_layers):
        super().__init__()

        out_dim = hid_dim
        self.n_layers = n_layers
        self.num_inits = int(layer.upper() == 'LSTM') + 1
        out_dim *= self.num_inits * n_layers

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim * self.n_layers)
        self.linear3 = nn.Linear(hid_dim * self.n_layers, out_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]

        out = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x)))))
        out = out.view(b, self.num_inits, self.n_layers, -1).permute(1, 2, 0, 3).contiguous()

        if self.num_inits == 2:
            return tuple([_ for _ in out])
        return out[0]


class Integrator(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel=1024):
        super().__init__()
        
        self.layer1 = nn.Linear(in_channel, hid_channel)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(hid_channel, hid_channel)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout(0.1)
        
        self.layer3 = nn.Linear(hid_channel, out_channel)
        
        
    def forward(self, x, feat):
        res = x
        mask = (feat != 0).all(dim=-1).all(dim=-1)
        
        out = torch.cat((x, feat), dim=-1)
        out = self.layer1(out)
        out = self.relu1(out)
        out = self.dr1(out)
        
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.dr2(out)
        
        out = self.layer3(out)
        out[mask] = out[mask] + res[mask]
        
        return out


class MotionEncoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 d_embed,
                 pose_dr,
                 rnn_type,
                 n_layers,
                 n_joints):
        super().__init__()
        
        self.n_joints = n_joints
        
        self.embed_layer = nn.Linear(in_dim, d_embed)
        self.pos_drop = nn.Dropout(pose_dr)
        
        # Keypoints initializer
        self.neural_init = NeuralInitialization(n_joints * 3 + in_dim, d_embed, rnn_type, n_layers)
        
        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)
        
    def forward(self, x, init):
        """ Forward pass of motion encoder.
        """
        
        self.b, self.f = x.shape[:2]
        x = self.embed_layer(x.reshape(self.b, self.f, -1))
        x = self.pos_drop(x)
        
        h0 = self.neural_init(init)
        pred_list = [init[..., :self.n_joints * 3]]
        motion_context_list = []
        
        for i in range(self.f):
            (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
            motion_context_list.append(motion_context)
            pred_list.append(pred_kp3d)
            
        pred_kp3d = torch.cat(pred_list[1:], dim=1).view(self.b, self.f, -1, 3)
        motion_context = torch.cat(motion_context_list, dim=1)
        
        # Merge 3D keypoints with motion context
        motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        return pred_kp3d, motion_context


class TrajectoryDecoder(nn.Module):
    def __init__(self, 
                 d_embed,
                 rnn_type,
                 n_layers):
        super().__init__()
        
        # Trajectory regressor
        self.regressor = Regressor(
            d_embed, d_embed, [3, 6], 12, rnn_type, n_layers, )
        
    def forward(self, x, root, cam_a, h0=None):
        """ Forward pass of trajectory decoder.
        """
        
        b, f = x.shape[:2]
        pred_root_list, pred_vel_list = [root[:, :1]], []
        
        for i in range(f):
            # Global coordinate estimation
            (pred_rootv, pred_rootr), _, h0 = self.regressor(
                x[:, [i]], [pred_root_list[-1], cam_a[:, [i]]], h0)
            
            pred_root_list.append(pred_rootr)
            pred_vel_list.append(pred_rootv)
        
        pred_root = torch.cat(pred_root_list, dim=1).view(b, f + 1, -1)
        pred_vel = torch.cat(pred_vel_list, dim=1).view(b, f, -1)
        
        return pred_root, pred_vel
        

class MotionDecoder(nn.Module):
    def __init__(self, 
                 d_embed,
                 rnn_type,
                 n_layers):
        super().__init__()
        
        self.n_pose = 24
        
        # SMPL pose initialization
        self.neural_init = NeuralInitialization(len(_C.BMODEL.MAIN_JOINTS) * 6, d_embed, rnn_type, n_layers)
        
        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [self.n_pose * 6, 10, 3, 4], self.n_pose * 6, rnn_type, n_layers)
        
    def forward(self, x, init):
        """ Forward pass of motion decoder.
        """
        b, f = x.shape[:2]
        
        h0 = self.neural_init(init[:, :, _C.BMODEL.MAIN_JOINTS].reshape(b, 1, -1))
        
        # Recursive prediction of SMPL parameters
        pred_pose_list = [init.reshape(b, 1, -1)]
        pred_shape_list, pred_cam_list, pred_contact_list = [], [], []
        
        for i in range(f):
            # Camera coordinate estimation
            (pred_pose, pred_shape, pred_cam, pred_contact), _, h0 = self.regressor(x[:, [i]], pred_pose_list[-1:], h0)
            pred_pose_list.append(pred_pose)
            pred_shape_list.append(pred_shape)
            pred_cam_list.append(pred_cam)
            pred_contact_list.append(pred_contact)
            
        pred_pose = torch.cat(pred_pose_list[1:], dim=1).view(b, f, -1)
        pred_shape = torch.cat(pred_shape_list, dim=1).view(b, f, -1)
        pred_cam = torch.cat(pred_cam_list, dim=1).view(b, f, -1)
        pred_contact = torch.cat(pred_contact_list, dim=1).view(b, f, -1)
        
        return pred_pose, pred_shape, pred_cam, pred_contact


class TrajectoryRefiner(nn.Module):
    def __init__(self, 
                 d_embed,
                 d_hidden, 
                 rnn_type,
                 n_layers):
        super().__init__()
        
        d_input = d_embed + 12
        self.refiner = Regressor(
            d_input, d_hidden, [6, 3], 9, rnn_type, n_layers)

    def forward(self, context, pred_vel, output, cam_angvel, return_y_up):
        b, f = context.shape[:2]
        
        # Register values
        pred_pose = output['pose'].clone().detach()
        pred_root = output['poses_root_r6d'].clone().detach()
        feet = output['feet'].clone().detach()
        contact = output['contact'].clone().detach()
        
        feet_vel = torch.cat((torch.zeros_like(feet[:, :1]), feet[:, 1:] - feet[:, :-1]), dim=1) * 30   # Normalize to 30 times
        feet = (feet_vel * contact.unsqueeze(-1)).reshape(b, f, -1)  # Velocity input
        inpt_feat = torch.cat([context, feet], dim=-1)
        
        (delta_root, delta_vel), _, _ = self.refiner(inpt_feat, [pred_root[:, 1:], pred_vel], h0=None)
        pred_root[:, 1:] = pred_root[:, 1:] + delta_root
        pred_vel = pred_vel + delta_vel

        root_world, trans_world = rollout_global_motion(pred_root, pred_vel)
        
        if return_y_up:
            yup2ydown = axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(root_world.device)
            root_world = yup2ydown.mT @ root_world
            trans_world = (yup2ydown.mT @ trans_world.unsqueeze(-1)).squeeze(-1)
            
        output.update({
            'poses_root_r6d_refined': pred_root,
            'vel_root_refined': pred_vel,
            'poses_root_world': root_world,
            'trans_world': trans_world,
        })
        
        return output