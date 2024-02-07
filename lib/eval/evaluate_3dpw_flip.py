import os
import os.path as osp
from glob import glob
from collections import defaultdict

import torch
import imageio
import numpy as np
from smplx import SMPL
from progress.bar import Bar


from configs import constants as _C
from configs.config import parse_args
from lib.utils.utils import create_logger
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.models.preproc.extractor import FeatureExtractor
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
from lib.utils import transforms
from lib.utils.imutils import avg_preds
from lib.utils.utils import prepare_output_dir
from lib.utils.utils import prepare_batch
from lib.models.smplify import TemporalSMPLify

try:
    from lib.vis.renderer import Renderer
    _render = True
except:
    print("PyTorch3D is not properly installed! Cannot render the SMPL mesh")
    _render = False


m2mm = 1e3
def main(cfg, args):
    logger = create_logger('output/evaluation', phase='3dpw_test')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(cfg, '3dpw', 'test', cfg.MODEL.BACKBONE)
    logger.info(f'Dataset loaded')
    
    # ========= Load feature extractor ========= #
    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
    extractor.flip_eval = True
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl_model = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl_model)
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
    
    for i in range(len(eval_loader)):
        # Original batch
        batch = eval_loader.dataset.load_data(i, False)
        x, inits, features, kwargs, gt = prepare_batch(batch, cfg.DEVICE, True)
        
        if cfg.FLIP_EVAL:
            flipped_batch = eval_loader.dataset.load_data(i, True)
            f_x, f_inits, f_features, f_kwargs, _ = prepare_batch(flipped_batch, cfg.DEVICE, True)
        
            # Forward pass with flipped input
            flipped_pred = network(f_x, f_inits, f_features, **f_kwargs)
        
        # Forward pass with normal input
        pred = network(x, inits, features, **kwargs)
        
        if cfg.FLIP_EVAL:
            # Merge two predictions
            flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
            pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
            flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
            avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
            avg_pose = avg_pose.reshape(-1, 144)

            # Refine trajectory with merged prediction
            network.pred_pose = avg_pose.view_as(network.pred_pose)
            network.pred_shape = avg_shape.view_as(network.pred_shape)
            output = network.forward_smpl(**kwargs)
            pred = network.refine_trajectory(output, return_y_up=True, **kwargs)
            
        if cfg.FIT_SMPLIFY:
        # if True:
            img_w, img_h = kwargs['res'][0].cpu().numpy()
            smplify = TemporalSMPLify(smpl_model, img_w=img_w, img_h=img_h, device=cfg.DEVICE)
            input_keypoints = batch['input_kp2d'][0].cpu().numpy()
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, batch['cam_angvel'], return_y_up=True)
        
        with torch.no_grad(): 
            # <======= Build predicted SMPL
            pred_output = smpl['neutral'](body_pose=pred['poses_body'], 
                                            global_orient=pred['poses_root_cam'], 
                                            betas=pred['betas'].squeeze(0), 
                                            pose2rot=False)
            pred_verts = pred_output.vertices.cpu()
            pred_j3d = torch.matmul(J_regressor_eval, pred_output.vertices).cpu()
            # =======>
            
            # <======= Build groundtruth SMPL
            target_output = smpl[batch['gender']](
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
            
            summary_string = (f'{batch["vid"]} | PA-MPJPE: {pa_mpjpe.mean():.1f}' + 
                              f'   MPJPE: {mpjpe.mean():.1f}' + 
                              f'   PVE: {pve.mean():.1f}' + 
                              f'   Accel: {accel.mean():.1f}')
            
            logger.info(summary_string)
            bar.suffix = summary_string
            bar.next()
            
            # <======= Accumulate the results over entire sequences
            accumulator['pa_mpjpe'].append(pa_mpjpe)
            accumulator['mpjpe'].append(mpjpe)
            accumulator['pve'].append(pve)
            accumulator['accel'].append(accel)
            # =======>
            
            # <======= (Optional) Render the prediction
            if not (_render and args.render):
                # Skip if PyTorch3D is not installed or rendering argument is not parsed.
                continue
            
                # Save path
            viz_pth = osp.join('output', 'visualization')
            os.makedirs(viz_pth, exist_ok=True)
            
            # Build Renderer
            width, height = batch['cam_intrinsics'][0][0, :2, -1].numpy() * 2
            focal_length = batch['cam_intrinsics'][0][0, 0, 0].item()
            renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl['neutral'].faces)
            
            # Get images and writer
            frame_list = batch['frame_id'][0].numpy()
            imname_list = sorted(glob(osp.join(_C.PATHS.THREEDPW_PTH, 'imageFiles', batch['vid'][:-2], '*.jpg')))
            writer = imageio.get_writer(osp.join(viz_pth, batch['vid'][0] + '.mp4'), 
                                        mode='I', format='FFMPEG', fps=30, macro_block_size=1)
            
            # Skip the invalid frames
            for i, frame in enumerate(frame_list):
                image = imageio.imread(imname_list[frame])
                
                # NOTE: pred['verts'] is different from pred_verts as we substracted offset from SMPL mesh.
                # Check line 121 in lib/models/smpl.py
                vertices = pred['verts_cam'][i] + pred['trans_cam'][[i]]
                image = renderer.render_mesh(vertices, image)
                writer.append_data(image)
            writer.close()
            # =======>
    
    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    print('')
    log_str = 'Evaluation on 3DPW, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
    logger.info(log_str)
    
if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg, args)