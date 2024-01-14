import os
import sys
import time
import colorsys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.models import build_network, build_body_model
from lib.models.preproc.slam import SLAMModel
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor


def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        visualize=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Preprocess
    if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
            osp.exists(osp.join(output_pth, 'slam_results.pth'))):
        
        detector = DetectionModel(cfg.DEVICE.lower())
        slam = SLAMModel(video, output_pth, width, height, calib)
        extractor = FeatureExtractor(cfg.DEVICE.lower())
        
        bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag: break
            
            # 2D detection and tracking
            detector.track(img, fps, length)
            
            # SLAM
            slam.track()
            
            bar.next()

        tracking_results = detector.process(fps)
        slam_results = slam.process()
    
        # Extract image features
        # TODO: Merge this into the previous while loop with an online bbox smoothing.
        tracking_results = extractor.run(video, tracking_results)
        logger.info('Complete Data preprocessing!')
        
        # Save the processed data
        joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
        joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Save processed data at {output_pth}')
    
    # If the processed data already exists, load the processed data
    else:
        tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
        slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    for batch in dataset:
        if batch is None: break
        
        # data
        _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
        
        # inference
        pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # Store results
        results[_id]['poses_body'] = pred['poses_body'].cpu().squeeze(0).numpy()
        results[_id]['poses_root_cam'] = pred['poses_root_cam'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts_cam'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['poses_root_world'] = pred['poses_root_world'].cpu().squeeze(0).numpy()
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['frame_id'] = frame_id
        
    # Visualizeimport pdb; pdb.set_trace()
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=True)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    with torch.no_grad():
        run(cfg, args.video, output_pth, network, args.calib, visualize=args.visualize)
        
    print()
    logger.info('Done !')