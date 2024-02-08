from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
from collections import defaultdict

import torch
import joblib
import numpy as np
from tqdm import tqdm
from smplx import SMPL

from configs import constants as _C
from lib.utils.data_utils import map_dmpl_to_smpl, transform_global_coordinate


@torch.no_grad()
def process_amass():
    target_fps = 30
    
    _, seqs, _ = next(os.walk(_C.PATHS.AMASS_PTH))
    
    zup2ydown = torch.Tensor(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    ).unsqueeze(0).float()
    
    smpl_dict = {'male': SMPL(model_path=_C.BMODEL.FLDR, gender='male'), 
                 'female': SMPL(model_path=_C.BMODEL.FLDR, gender='female'),
                 'neutral': SMPL(model_path=_C.BMODEL.FLDR)}
    processed_data = defaultdict(list)
    
    for seq in (seq_bar := tqdm(sorted(seqs), leave=True)):
        seq_bar.set_description(f'Dataset: {seq}')
        seq_fldr = osp.join(_C.PATHS.AMASS_PTH, seq)
        _, subjs, _ = next(os.walk(seq_fldr))
        
        for subj in (subj_bar := tqdm(sorted(subjs), leave=False)):
            subj_bar.set_description(f'Subject: {subj}')
            subj_fldr = osp.join(seq_fldr, subj)
            acts = [x for x in os.listdir(subj_fldr) if x.endswith('.npz')]
            
            for act in (act_bar := tqdm(sorted(acts), leave=False)):
                act_bar.set_description(f'Action: {act}')
                
                # Load data
                fname = osp.join(subj_fldr, act)
                if fname.endswith('shape.npz') or fname.endswith('stagei.npz'): 
                    # Skip shape and stagei files
                    continue
                data = dict(np.load(fname, allow_pickle=True))
                
                # Resample data to target_fps
                key = [k for k in data.keys() if 'mocap_frame' in k][0]
                mocap_framerate = data[key]
                retain_freq = int(mocap_framerate / target_fps + 0.5)
                num_frames = len(data['poses'][::retain_freq])
                
                # Skip if the sequence is too short
                if num_frames < 25: continue
                
                # Get SMPL groundtruth from MoSh fitting
                pose = map_dmpl_to_smpl(torch.from_numpy(data['poses'][::retain_freq]).float())
                transl = torch.from_numpy(data['trans'][::retain_freq]).float()
                betas = torch.from_numpy(
                    np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)).float()
                
                # Convert Z-up coordinate to Y-down
                pose, transl = transform_global_coordinate(pose, zup2ydown, transl)
                pose = pose.reshape(-1, 72)
                
                # Create SMPL mesh
                gender = str(data['gender'])
                if not gender in ['male', 'female', 'neutral']: 
                    if 'female' in gender: gender = 'female'
                    elif 'neutral' in gender: gender = 'neutral'
                    elif 'male' in gender: gender = 'male'
                
                output = smpl_dict[gender](body_pose=pose[:, 3:], 
                                            global_orient=pose[:, :3], 
                                            betas=betas,
                                            transl=transl)
                vertices = output.vertices
                
                # Assume motion starts with 0-height
                init_height = vertices[0].max(0)[0][1]
                transl[:, 1] = transl[:, 1] + init_height
                vertices[:, :, 1] = vertices[:, :, 1] - init_height
                
                # Append data
                processed_data['pose'].append(pose.numpy())
                processed_data['betas'].append(betas.numpy())
                processed_data['transl'].append(transl.numpy())
                processed_data['vid'].append(np.array([f'{seq}_{subj}_{act}'] * pose.shape[0]))

    for key, val in processed_data.items():
        processed_data[key] = np.concatenate(val)

    joblib.dump(processed_data, _C.PATHS.AMASS_LABEL)
    print('\nDone!')

if __name__ == '__main__':
    out_path = '/'.join(_C.PATHS.AMASS_LABEL.split('/')[:-1])
    os.makedirs(out_path, exist_ok=True)
    
    process_amass()