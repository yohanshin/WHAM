import os
import torch
from tqdm import tqdm

from lib.models import build_body_model
from .losses import SMPLifyLoss

class TemporalSMPLify():
    
    def __init__(self, 
                 smpl=None,
                 lr=1e-2,
                 num_iters=5,
                 num_steps=10,
                 img_w=None,
                 img_h=None,
                 device=None
                 ):
        
        self.smpl = smpl
        self.lr = lr
        self.num_iters = num_iters
        self.num_steps = num_steps
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        
    def fit(self, init_pred, keypoints, bbox, **kwargs):
        
        def to_params(param):
            return torch.from_numpy(param).float().to(self.device).requires_grad_(True)
        
        pose = init_pred['pose'].detach().cpu().numpy()
        betas = init_pred['betas'].detach().cpu().numpy()
        cam = init_pred['cam'].detach().cpu().numpy()
        keypoints = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)
        
        BN = pose.shape[1]
        lr = self.lr
        
        # Stage 1. Optimize translation
        params = [to_params(pose), to_params(betas), to_params(cam)]
        optim_params = [params[2]]
        
        optimizer = torch.optim.LBFGS(
            optim_params, 
            lr=lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        loss_fn = SMPLifyLoss(init_pose=pose, device=self.device, **kwargs)
        
        closure = loss_fn.create_closure(optimizer,
                       self.smpl, 
                       params,
                       bbox,
                       keypoints)
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
                
        
        # Stage 2. Optimize all params
        optimizer = torch.optim.LBFGS(
            params, 
            lr=lr * BN, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
        
        init_pred['pose'] = params[0].detach()
        init_pred['betas'] = params[1].detach()
        init_pred['cam'] = params[2].detach()
        
        return init_pred