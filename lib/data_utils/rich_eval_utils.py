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
detection_results_dir = 'dataset/detection_results/RICH'

def extract_cam_param_xml(xml_path='', dtype=torch.float32):
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    # distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center = torch.tensor([cam_center], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center

@torch.no_grad()
def preprocess(dset, batch_size):
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, choices=['1', '2'], help='Data split')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Data split')
    args = parser.parse_args()
    
    preprocess(args.split, args.batch_size)