from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown, _track_by_iou


def get_track_id(results, results_last, next_id, min_keypoints=3, tracking_thr=0.3):
    for result in results:
        track_id, results_last, match_result = _track_by_iou(result, results_last,
                                                      tracking_thr)
        if track_id == -1:
            if np.count_nonzero(result.pred_instances.keypoints[0, :, 1]) > min_keypoints:
                result.track_id = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                result.pred_instances.keypoints[0, :, :] = -10
                result.pred_instances.bboxes *= 0
                result.track_id = -1
        else:
            result.track_id = track_id
        del match_result

    return results, next_id


class DetectionModel(object):
    def __init__(self, device, mmpose_cfg):
        
        # ViTPose
        self.mmpose_cfg = mmpose_cfg
        pose_model_cfg = mmpose_cfg.POSE_CONFIG
        pose_model_ckpt = mmpose_cfg.POSE_CHECKPOINT
        self.pose_model = init_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        
        # YOLO
        bbox_model_ckpt = mmpose_cfg.DET_CHECKPOINT
        self.bbox_model = YOLO(bbox_model_ckpt)
        
        self.device = device
        self.track_thr = self.mmpose_cfg.TRACKING_THR
        self.min_frames = self.mmpose_cfg.MINIMUM_FRMAES
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
    def xyxy_to_cxcys(self, bbox):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        return np.array([[cx, cy, scale]])
        
    def detect(self, img):
        
        # bbox detection, output list of np.array(float32)
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=self.mmpose_cfg.BBOX_CONF, save=False, verbose=False)[0].boxes.xyxy.detach().cpu().numpy()
        
        
        pose_results = inference_topdown(
            self.pose_model,
            img,
            bboxes=bboxes,
            bbox_format='xyxy')
        return bboxes, pose_results
    
    def track_detections(self, pose_results):
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            tracking_thr=self.track_thr)
        for pose_result in pose_results:
            _id = pose_result.track_id
            xyxy = pose_result.pred_instances.bboxes[0]
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            kpts = np.zeros((1,17,3), dtype=float)
            kpts[:,:,:2] = pose_result.pred_instances.keypoints
            kpts[:,:,2] = pose_result.pred_instances.keypoint_scores
            self.tracking_results['keypoints'].append(kpts)
        
        self.frame_id += 1
        self.pose_results_last = pose_results

    def track(self, img, fps, length):
        bboxes, pose_results = self.detect(img)
        self.track_detections(pose_results)
        
    def process(self, fps):
        for key in ['id', 'frame_id']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        for key in ['keypoints', 'bbox']:
            self.tracking_results[key] = np.concatenate(self.tracking_results[key], axis=0)
            
        output = defaultdict(dict)
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            output[_id]['features'] = []
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < self.min_frames:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output
    

class TrackingModel(DetectionModel):
    def __init__(self, track_thr=.3, min_frames=10):
        self.track_thr = track_thr
        self.min_frames = min_frames
        self.initialize_tracking()
