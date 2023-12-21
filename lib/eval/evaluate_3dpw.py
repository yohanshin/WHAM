import os
import time
import os.path as osp
from collections import defaultdict

import torch
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from configs.config import parse_args
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
from lib.utils import transforms
from lib.utils.utils import prepare_output_dir
from lib.utils.utils import prepare_batch

m2mm = 1e3
@torch.no_grad()
def main(cfg, args):
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(cfg, '3dpw', 'test', cfg.MODEL.BACKBONE)
    logger.info(f'Dataset loaded')
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Build SMPL models with each gender
    smpl = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE) for k in ['male', 'female', 'neutral']}
    
    # Load vertices -> joints regression matrix to evaluate
    J_regressor_eval = torch.from_numpy(
        np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
    )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(cfg.DEVICE)
    pelvis_idxs = [2, 3]
    
    accumulator = defaultdict(list)
    bar = Bar('Inference', fill='#', max=len(eval_loader))
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            x, inits, features, gt = prepare_batch(batch, cfg.DEVICE, True)
            
            # <======= Inference
            pred = network(x, inits, features, **gt)
            # =======>
            
            # <======= Build predicted SMPL
            pred_output = smpl['neutral'](body_pose=pred['poses_body'], 
                                          global_orient=pred['poses_root_cam'], 
                                          betas=pred['betas'].squeeze(0), 
                                          pose2rot=False)
            pred_verts = pred_output.vertices.cpu()
            pred_j3d = torch.matmul(J_regressor_eval, pred_output.vertices).cpu()
            # =======>
            
            # <======= Build groundtruth SMPL
            target_output = smpl[batch['gender'][0]](
                body_pose=transforms.rotation_6d_to_matrix(gt['pose'][0, :, 1:]),
                global_orient=transforms.rotation_6d_to_matrix(gt['pose'][0, :, :1]),
                betas=gt['betas'][0],
                pose2rot=False)
            target_verts = target_output.vertices.cpu()
            target_j3d = torch.matmul(J_regressor_eval, target_output.vertices).cpu()
            # =======>
            
            # <======= Compute performance of the current sequence
            pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
                [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs
            )
            S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
            pa_mpjpe = torch.sqrt(((S1_hat - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
            mpjpe = torch.sqrt(((pred_j3d - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
            pve = torch.sqrt(((pred_verts - target_verts) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
            accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d)[1:-1]
            accel = accel * (30 ** 2)       # per frame^s to per s^2
            # =======>
            
            summary_string = f'{batch["vid"][0]} | PA-MPJPE: {pa_mpjpe.mean():.1f}   MPJPE: {mpjpe.mean():.1f}   PVE: {pve.mean():.1f}'
            bar.suffix = summary_string
            bar.next()
            
            # <======= Accumulate the results over entire sequences
            accumulator['pa_mpjpe'].append(pa_mpjpe)
            accumulator['mpjpe'].append(mpjpe)
            accumulator['pve'].append(pve)
            accumulator['accel'].append(accel)
            # =======>
            
    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    log_str = 'Evaluation on 3DPW, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
    logger.info(log_str)
    
if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg, args)