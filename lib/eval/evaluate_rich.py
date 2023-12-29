import os
import time
import os.path as osp
from collections import defaultdict

import torch
import joblib
import numpy as np
from loguru import logger
from smplx import SMPL, SMPLX
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
smplx2smpl = torch.from_numpy(joblib.load(_C.BMODEL.SMPLX2SMPL)['matrix']).unsqueeze(0).float().cuda()
@torch.no_grad()
def main(cfg, args):
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(cfg, 'rich', 'test', cfg.MODEL.BACKBONE)
    logger.info(f'Dataset loaded')
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Build neutral SMPL model for WHAM and gendered SMPLX models for the groundtruth data
    smpl = SMPL(_C.BMODEL.FLDR, gender='neutral').to(cfg.DEVICE)
    
    # Load vertices -> joints regression matrix to evaluate
    J_regressor_eval = smpl.J_regressor.clone().unsqueeze(0)
    pelvis_idxs = [1, 2]
    
    accumulator = defaultdict(list)
    bar = Bar('Inference', fill='#', max=len(eval_loader))
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            x, inits, features, gt = prepare_batch(batch, cfg.DEVICE, True)
            
            # <======= Inference
            pred = network(x, inits, features, **gt)
            # =======>
            
            # <======= Build predicted SMPL
            pred_output = smpl(body_pose=pred['poses_body'], 
                               global_orient=pred['poses_root_cam'], 
                               betas=pred['betas'].squeeze(0), 
                               pose2rot=False)
            pred_verts = pred_output.vertices.cpu()
            pred_j3d = torch.matmul(J_regressor_eval, pred_output.vertices).cpu()
            # =======>
            
            # <======= Build groundtruth SMPL (from SMPLX)
            smplx = SMPLX(_C.BMODEL.FLDR.replace('smpl', 'smplx'), 
                          gender=batch['gender'][0], 
                          batch_size=len(pred_verts)
            ).to(cfg.DEVICE)
            gt_pose = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(gt['pose'][0]))
            target_output = smplx(
                body_pose=gt_pose[:, 1:-2].reshape(-1, 63),
                global_orient=gt_pose[:, 0],
                betas=gt['betas'][0])
            target_verts = torch.matmul(smplx2smpl, target_output.vertices.cuda()).cpu()
            target_j3d = torch.matmul(J_regressor_eval, target_verts.to(cfg.DEVICE)).cpu()
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

    print('')
    log_str = 'Evaluation on RICH, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
    logger.info(log_str)
    
if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg, args)