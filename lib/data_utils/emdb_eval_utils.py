from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import pickle
import joblib
import argparse
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from lib.models.smpl import SMPL
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.preproc.backbone.utils import process_image
from lib.utils import transforms
from lib.utils.imutils import (
    flip_kp, flip_bbox
)

dataset = defaultdict(list)
detection_results_dir = 'dataset/detection_results/EMDB'

def is_dset(emdb_pkl_file, dset):
    target_dset = 'emdb' + dset
    with open(emdb_pkl_file, "rb") as f:
        data = pickle.load(f)
        return data[target_dset]
    
@torch.no_grad()
def preprocess(dset, batch_size):
    
    tt = lambda x: torch.from_numpy(x).float()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_pth = osp.join(_C.PATHS.PARSED_DATA, f'emdb_{dset}_vit.pth') # Use ViT feature extractor
    extractor = FeatureExtractor(device, flip_eval=True, max_batch_size=batch_size)
    
    all_emdb_pkl_files = sorted(glob(os.path.join(_C.PATHS.EMDB_PTH, "*/*/*_data.pkl")))
    emdb_sequence_roots = []
    both = []
    for emdb_pkl_file in all_emdb_pkl_files:
        if is_dset(emdb_pkl_file, dset):
            emdb_sequence_roots.append(os.path.dirname(emdb_pkl_file))
    
    smpl = {
        'neutral': SMPL(model_path=_C.BMODEL.FLDR),
        'male': SMPL(model_path=_C.BMODEL.FLDR, gender='male'),
        'female': SMPL(model_path=_C.BMODEL.FLDR, gender='female'),
    }
    
    for sequence in emdb_sequence_roots:
        subj, seq = sequence.split('/')[-2:]
        annot_pth = glob(osp.join(sequence, '*_data.pkl'))[0]
        annot = pickle.load(open(annot_pth, 'rb'))
        
        # Get ground truth data
        gender = annot['gender']
        masks = annot['good_frames_mask']
        poses_body = annot["smpl"]["poses_body"]
        poses_root = annot["smpl"]["poses_root"]
        betas = np.repeat(annot["smpl"]["betas"].reshape((1, -1)), repeats=annot["n_frames"], axis=0)
        extrinsics = annot["camera"]["extrinsics"]
        width, height = annot['camera']['width'], annot['camera']['height']
        xyxys = annot['bboxes']['bboxes']
        
        # Map to camear coordinate
        poses_root_cam = transforms.matrix_to_axis_angle(tt(extrinsics[:, :3, :3]) @ transforms.axis_angle_to_matrix(tt(poses_root)))
        poses = np.concatenate([poses_root_cam, poses_body], axis=-1)
        
        pred_kp2d = np.load(osp.join(detection_results_dir, f'{subj}_{seq}.npy'))
        
        # ======== Extract features ======== #
        imname_list = sorted(glob(osp.join(sequence, 'images/*')))
        bboxes, frame_ids, patch_list, features, flipped_features = [], [], [], [], []
        bar = Bar(f'Load images', fill='#', max=len(imname_list))
        for idx, (imname, xyxy, mask) in enumerate(zip(imname_list, xyxys, masks)):
            if not mask: continue
            
            # ========= Load image ========= #
            img_rgb = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2RGB)
            
            # ========= Load bbox ========= #
            x1, y1, x2, y2 = xyxy
            bbox = np.array([(x1 + x2)/2., (y1 + y2)/2., max(x2 - x1, y2 - y1) / 1.1])
            
            # ========= Process image ========= #
            norm_img, crop_img = process_image(img_rgb, bbox[:2], bbox[2] / 200, 256, 256)
            
            patch_list.append(torch.from_numpy(norm_img).unsqueeze(0).float())
            bboxes.append(bbox)
            frame_ids.append(idx)
            bar.next()
        
        patch_list = torch.split(torch.cat(patch_list), batch_size)
        bboxes = torch.from_numpy(np.stack(bboxes)).float()
        for i, patch in enumerate(patch_list):
            bbox = bboxes[i*batch_size:min((i+1)*batch_size, len(frame_ids))].float().cuda()
            bbox_center = bbox[:, :2]
            bbox_scale = bbox[:, 2] / 200

            feature = extractor.model(patch.cuda(), encode=True)
            features.append(feature.cpu())
            
            flipped_feature = extractor.model(torch.flip(patch, (3, )).cuda(), encode=True)
            flipped_features.append(flipped_feature.cpu())
            
            if i == 0:
                init_patch = patch[[0]].clone()
        
        features = torch.cat(features)
        flipped_features = torch.cat(flipped_features)
        res_h, res_w = img_rgb.shape[:2]
    
        # ======== Append data ======== #
        dataset['gender'].append(gender)
        dataset['bbox'].append(bboxes)
        dataset['res'].append(torch.tensor([[width, height]]).repeat(len(frame_ids), 1).float())
        dataset['vid'].append(f'{subj}_{seq}')
        dataset['pose'].append(tt(poses)[frame_ids])
        dataset['betas'].append(tt(betas)[frame_ids])
        dataset['kp2d'].append(tt(pred_kp2d)[frame_ids])
        dataset['frame_id'].append(torch.from_numpy(np.array(frame_ids)))
        dataset['cam_poses'].append(tt(extrinsics)[frame_ids])
        dataset['features'].append(features)
        dataset['flipped_features'].append(flipped_features)
        
        # Flipped data
        dataset['flipped_bbox'].append(
            torch.from_numpy(flip_bbox(dataset['bbox'][-1].clone().numpy(), res_w, res_h)).float()
        )
        dataset['flipped_kp2d'].append(
            torch.from_numpy(flip_kp(dataset['kp2d'][-1].clone().numpy(), res_w)).float()
        )
        # ======== Append data ======== #
        
        # Pad 1 frame
        for key, val in dataset.items():
            if isinstance(val[-1], torch.Tensor):
                dataset[key][-1] = torch.cat((val[-1][:1].clone(), val[-1][:]), dim=0)
        
        # Initial predictions
        bbox = bboxes[:1].clone().cuda()
        bbox_center = bbox[:, :2].clone()
        bbox_scale = bbox[:, 2].clone() / 200
        kwargs = {'img_w': torch.tensor(res_w).repeat(1).float().cuda(), 
                    'img_h': torch.tensor(res_h).repeat(1).float().cuda(), 
                    'bbox_center': bbox_center, 'bbox_scale': bbox_scale}

        pred_global_orient, pred_pose, pred_shape, _ = extractor.model(init_patch.cuda(), **kwargs)
        pred_output = smpl['neutral'].get_output(global_orient=pred_global_orient.cpu(),
                                                body_pose=pred_pose.cpu(),
                                                betas=pred_shape.cpu(),
                                                pose2rot=False)
        init_kp3d = pred_output.joints
        init_pose = transforms.matrix_to_axis_angle(torch.cat((pred_global_orient, pred_pose), dim=1))
        
        dataset['init_kp3d'].append(init_kp3d)
        dataset['init_pose'].append(init_pose.cpu())
        
        # Flipped initial predictions
        bbox_center[:, 0] = res_w - bbox_center[:, 0]
        pred_global_orient, pred_pose, pred_shape, _ = extractor.model(torch.flip(init_patch, (3, )).cuda(), **kwargs)
        pred_output = smpl['neutral'].get_output(global_orient=pred_global_orient.cpu(),
                                                body_pose=pred_pose.cpu(),
                                                betas=pred_shape.cpu(),
                                                pose2rot=False)
        init_kp3d = pred_output.joints
        init_pose = transforms.matrix_to_axis_angle(torch.cat((pred_global_orient, pred_pose), dim=1))
        
        dataset['flipped_init_kp3d'].append(init_kp3d)
        dataset['flipped_init_pose'].append(init_pose.cpu())
        
    joblib.dump(dataset, save_pth)
    logger.info(f'==> Done !')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, choices=['1', '2'], help='Data split')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Data split')
    args = parser.parse_args()
    
    preprocess(args.split, args.batch_size)