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

dataset = defaultdict(list)
detection_results_dir = 'dataset/detection_results/3DPW'
tcmr_annot_pth = 'dataset/parsed_data/TCMR_preproc/3dpw_train_db.pt'


@torch.no_grad()
def preprocess(batch_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_pth = osp.join(_C.PATHS.PARSED_DATA, f'3pdw_train_vit.pth') # Use ViT feature extractor
    extractor = FeatureExtractor(device, flip_eval=True, max_batch_size=batch_size)
    
    tcmr_data = joblib.load(tcmr_annot_pth)
    
    annot_file_list, idxs = np.unique(tcmr_data['vid_name'], return_index=True)
    idxs = idxs.tolist()
    annot_file_list = [annot_file_list[idxs.index(idx)] for idx in sorted(idxs)]
    annot_file_list = [osp.join(_C.PATHS.THREEDPW_PTH, 'sequenceFiles', 'train', annot_file[:-2] + '.pkl') for annot_file in annot_file_list]
    annot_file_list = list(dict.fromkeys(annot_file_list))
    
    vid_idx = 0
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
            smpl_gender = SMPL(model_path=_C.BMODEL.FLDR, gender=gender)
            
            # ======== Add TCMR data ======== #
            vid_name = f'{seq}_{p_id}'
            tcmr_ids = [i for i, v in enumerate(tcmr_data['vid_name']) if vid_name in v]
            frame_ids = tcmr_data['frame_id'][tcmr_ids]
            
            pose = torch.from_numpy(data['poses'][p_id]).float()[frame_ids]
            shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
            trans = torch.from_numpy(data['trans'][p_id]).float()[frame_ids]
            cam_poses = torch.from_numpy(data['cam_poses'][frame_ids]).float()
            
            # ======== Align the mesh params ======== #
            Rc = cam_poses[:, :3, :3]
            Tc = cam_poses[:, :3, 3]
            org_output = smpl_gender.get_output(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=trans)
            org_v0 = (org_output.vertices + org_output.offset.unsqueeze(1)).mean(1)
            pose = torch.from_numpy(tcmr_data['pose'][tcmr_ids]).float()
            
            output = smpl_gender.get_output(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3])
            v0 = (output.vertices + output.offset.unsqueeze(1)).mean(1)
            trans = (Rc @ org_v0.reshape(-1, 3, 1)).reshape(-1, 3) + Tc - v0
            j3d = output.joints + (output.offset + trans).unsqueeze(1)
            j2d = torch.div(j3d, j3d[..., 2:])
            kp2d = torch.matmul(K, j2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
            # ======== Align the mesh params ======== #
            
            # ======== Get detection results ======== #
            fname = f'{seq}_{p_id}.npy'
            pred_kp2d = torch.from_numpy(
                np.load(osp.join(detection_results_dir, fname))
            ).float()[frame_ids]
            # ======== Get detection results ======== #
            
            img_paths = sorted(glob(osp.join(_C.PATHS.THREEDPW_PTH, 'imageFiles', seq, '*.jpg')))
            img_paths = [img_path for i, img_path in enumerate(img_paths) if i in frame_ids]
            img = cv2.imread(img_paths[0]); res_h, res_w = img.shape[:2]
            vid_idxs = torch.from_numpy(np.array([vid_idx] * len(img_paths)).astype(int))
            vid_idx += 1
            
            # ======== Append data ======== #
            dataset['bbox'].append(torch.from_numpy(tcmr_data['bbox'][tcmr_ids].copy()).float())
            dataset['res'].append(torch.tensor([[res_w, res_h]]).repeat(len(frame_ids), 1).float())
            dataset['vid'].append(vid_idxs)
            dataset['pose'].append(pose)
            dataset['betas'].append(shape)
            dataset['transl'].append(trans)
            dataset['kp2d'].append(pred_kp2d)
            dataset['joints3D'].append(j3d)
            dataset['joints2D'].append(kp2d)
            dataset['frame_id'].append(torch.from_numpy(frame_ids))
            dataset['cam_poses'].append(cam_poses)
            dataset['gender'].append(torch.tensor([['male','female'].index(gender)]).repeat(len(frame_ids)))
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
            features = []
            for i, patch in enumerate(patch_list):
                pred = extractor.model(patch.cuda(), encode=True)
                features.append(pred.cpu())

            features = torch.cat(features)
            dataset['features'].append(features)
            # ======== Extract features ======== #
            
    for key in dataset.keys():
        dataset[key] = torch.cat(dataset[key])
        
    joblib.dump(dataset, save_pth)
    logger.info(f'\n ==> Done !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Data split')
    args = parser.parse_args()
    
    preprocess(args.batch_size)