from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

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
detection_results_dir = 'dataset/detection_results/3DPW'
tcmr_annot_pth = 'dataset/parsed_data/TCMR_preproc/3dpw_dset_db.pt'

@torch.no_grad()
def preprocess(dset, batch_size):

    if dset == 'val': _dset = 'validation'
    else: _dset = dset
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_pth = osp.join(_C.PATHS.PARSED_DATA, f'3pdw_{dset}_vit.pth') # Use ViT feature extractor
    extractor = FeatureExtractor(device, flip_eval=True, max_batch_size=batch_size)
    
    tcmr_data = joblib.load(tcmr_annot_pth.replace('dset', dset))
    smpl_neutral = SMPL(model_path=_C.BMODEL.FLDR)
    
    annot_file_list, idxs = np.unique(tcmr_data['vid_name'], return_index=True)
    idxs = idxs.tolist()
    annot_file_list = [annot_file_list[idxs.index(idx)] for idx in sorted(idxs)]
    annot_file_list = [osp.join(_C.PATHS.THREEDPW_PTH, 'sequenceFiles', _dset, annot_file[:-2] + '.pkl') for annot_file in annot_file_list]
    annot_file_list = list(dict.fromkeys(annot_file_list))
    
    for annot_file in annot_file_list:
        seq = annot_file.split('/')[-1].split('.')[0]
        
        data = pickle.load(open(annot_file, 'rb'), encoding='latin1')
        
        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)
        
        K = torch.from_numpy(data['cam_intrinsics']).unsqueeze(0).float()
        
        for p_id in range(num_people):

            logger.info(f'==> {seq} {p_id}')
            gender = {'m': 'male', 'f': 'female'}[data['genders'][p_id]]
            
            # ======== Add TCMR data ======== #
            vid_name = f'{seq}_{p_id}'
            tcmr_ids = [i for i, v in enumerate(tcmr_data['vid_name']) if vid_name in v]
            frame_ids = tcmr_data['frame_id'][tcmr_ids]
            
            pose = torch.from_numpy(data['poses'][p_id]).float()[frame_ids]
            shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
            pose = torch.from_numpy(tcmr_data['pose'][tcmr_ids]).float()    # Camera coordinate
            cam_poses = torch.from_numpy(data['cam_poses'][frame_ids]).float()
            
            # ======== Get detection results ======== #
            fname = f'{seq}_{p_id}.npy'
            pred_kp2d = torch.from_numpy(
                np.load(osp.join(detection_results_dir, fname))
            ).float()[frame_ids]
            # ======== Get detection results ======== #
            
            img_paths = sorted(glob(osp.join(_C.PATHS.THREEDPW_PTH, 'imageFiles', seq, '*.jpg')))
            img_paths = [img_path for i, img_path in enumerate(img_paths) if i in frame_ids]
            img = cv2.imread(img_paths[0]); res_h, res_w = img.shape[:2]
            vid_idxs = fname.split('.')[0]
        
            # ======== Append data ======== #
            dataset['gender'].append(gender)
            dataset['vid'].append(vid_idxs)
            dataset['pose'].append(pose)
            dataset['betas'].append(shape)
            dataset['cam_poses'].append(cam_poses)
            dataset['frame_id'].append(torch.from_numpy(frame_ids))
            dataset['res'].append(torch.tensor([[res_w, res_h]]).repeat(len(frame_ids), 1).float())
            dataset['bbox'].append(torch.from_numpy(tcmr_data['bbox'][tcmr_ids].copy()).float())
            dataset['kp2d'].append(pred_kp2d)
            
            # Flipped data
            dataset['flipped_bbox'].append(
                torch.from_numpy(flip_bbox(dataset['bbox'][-1].clone().numpy(), res_w, res_h)).float()
            )
            dataset['flipped_kp2d'].append(
                torch.from_numpy(flip_kp(dataset['kp2d'][-1].clone().numpy(), res_w)).float()
            )
            # ======== Append data ======== #            
            
            # ======== Extract features ======== #
            patch_list = []
            bboxes = dataset['bbox'][-1].clone().numpy()
            bar = Bar(f'Load images', fill='#', max=len(img_paths))
            
            for img_path, bbox in zip(img_paths, bboxes):
                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                norm_img, crop_img = process_image(img_rgb, bbox[:2], bbox[2] / 200, 256, 256)
                patch_list.append(torch.from_numpy(norm_img).unsqueeze(0).float())
                bar.next()

            patch_list = torch.split(torch.cat(patch_list), batch_size)
            features, flipped_features = [], []
            for i, patch in enumerate(patch_list):
                feature = extractor.model(patch.cuda(), encode=True)
                features.append(feature.cpu())
                
                flipped_feature = extractor.model(torch.flip(patch, (3, )).cuda(), encode=True)
                flipped_features.append(flipped_feature.cpu())
                
                if i == 0:
                    init_patch = patch[[0]].clone()

            features = torch.cat(features)
            flipped_features = torch.cat(flipped_features)
            dataset['features'].append(features)
            dataset['flipped_features'].append(flipped_features)
            # ======== Extract features ======== #
            
            # Pad 1 frame
            for key, val in dataset.items():
                if isinstance(val[-1], torch.Tensor):
                    dataset[key][-1] = torch.cat((val[-1][:1].clone(), val[-1][:]), dim=0)
            
            # Initial predictions
            bbox = torch.from_numpy(bboxes[:1].copy()).float().cuda()
            bbox_center = bbox[:, :2].clone()
            bbox_scale = bbox[:, 2].clone() / 200
            kwargs = {'img_w': torch.tensor(res_w).repeat(1).float().cuda(), 
                      'img_h': torch.tensor(res_h).repeat(1).float().cuda(), 
                     'bbox_center': bbox_center, 'bbox_scale': bbox_scale}

            pred_global_orient, pred_pose, pred_shape, _ = extractor.model(init_patch.cuda(), **kwargs)
            pred_output = smpl_neutral.get_output(global_orient=pred_global_orient.cpu(),
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
            pred_output = smpl_neutral.get_output(global_orient=pred_global_orient.cpu(),
                                                  body_pose=pred_pose.cpu(),
                                                  betas=pred_shape.cpu(),
                                                  pose2rot=False)
            init_kp3d = pred_output.joints
            init_pose = transforms.matrix_to_axis_angle(torch.cat((pred_global_orient, pred_pose), dim=1))
            
            dataset['flipped_init_kp3d'].append(init_kp3d)
            dataset['flipped_init_pose'].append(init_pose.cpu())
            
    joblib.dump(dataset, save_pth)
    logger.info(f'\n ==> Done !')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, choices=['val', 'test'], help='Data split')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Data split')
    args = parser.parse_args()
    
    preprocess(args.split, args.batch_size)