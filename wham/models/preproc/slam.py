import cv2
import numpy as np
import glob
import os.path as osp
import os
import time
import torch
from pathlib import Path
from multiprocessing import Process, Queue

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import video_stream


class SLAMModel(object):
    def __init__(self, cfg, video, output_pth, width, height, calib=None, stride=1, skip=0, buffer=2048):
        if (calib is None) or not osp.exists(calib): 
            calib = osp.join(output_pth, 'calib.txt')
        if not osp.exists(calib):
            self.estimate_intrinsics(width, height, calib)
        
        self.dpvo_cfg = cfg.CFG
        self.dpvo_ckpt = cfg.CKPT
        
        self.buffer = buffer
        self.times = []
        self.slam = None
        self.queue = Queue(maxsize=8)
        self.reader = Process(target=video_stream, args=(self.queue, video, calib, stride, skip))
        self.reader.start()
        
    def estimate_intrinsics(self, width, height, calib):
        focal_length = (height ** 2 + width ** 2) ** 0.5
        center_x = width / 2
        center_y = height / 2
        
        with open(calib, 'w') as fopen:
            line = f'{focal_length} {focal_length} {center_x} {center_y}'
            fopen.write(line)
            
            
    def track(self, ):
        (t, image, intrinsics) = self.queue.get()
        
        if t < 0: return
        
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()
        
        if self.slam is None:
            cfg.merge_from_file(self.dpvo_cfg)
            cfg.BUFFER_SIZE = self.buffer
            self.slam = DPVO(cfg, self.dpvo_ckpt, ht=image.shape[1], wd=image.shape[2], viz=False)
        
        with Timer("SLAM", enabled=False):
            t = time.time()
            self.slam(t, image, intrinsics)
            self.times.append(time.time() - t)
            
            
    def process(self, ):
        for _ in range(12):
            self.slam.update()
        
        self.reader.join()
        return self.slam.terminate()[0]