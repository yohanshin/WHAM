import os
from yacs.config import CfgNode as CN

get_env_var = os.environ.get

cfg = CN()

cfg.LOGDIR=''
cfg.DEVICE='cuda'
cfg.EXP_NAME='demo'
cfg.OUTPUT_DIR='experiments/'
cfg.NUM_WORKERS=0
cfg.MODEL_CONFIG=get_env_var('WHAM_CFG', './configs/yamls/model_base.yaml')

cfg.TRAIN = CN()
cfg.TRAIN.STAGE = 'stage2'
cfg.TRAIN.CHECKPOINT = get_env_var('WHAM_CKPT', './checkpoints/wham_vit_bedlam_w_3dpw.pth.tar')

cfg.MODEL = CN()
cfg.MODEL.BACKBONE = 'vit'

cfg.MMPOSE_CFG = CN()
cfg.MMPOSE_CFG.POSE_CONFIG = get_env_var('POSE2D_CFG', './configs/VIT/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py')
cfg.MMPOSE_CFG.POSE_CHECKPOINT = get_env_var('POSE2D_CKPT', 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth')
cfg.MMPOSE_CFG.DET_CONFIG = get_env_var('DETECTOR_CFG', './configs/rtmdet_m_8xb32-300e_coco.py')
cfg.MMPOSE_CFG.DET_CHECKPOINT = get_env_var('DETECTOR_CKPT', './checkpoints/yolov8m.pt')
cfg.MMPOSE_CFG.BBOX_CONF = 0.5
cfg.MMPOSE_CFG.TRACKING_THR = 0.1
cfg.MMPOSE_CFG.MINIMUM_FRMAES = 30

cfg.DPVO = CN()
cfg.DPVO.CFG = get_env_var('DPVO_CFG', './configs/DPVO/default.yaml')
cfg.DPVO.CKPT = get_env_var('DPVO_CKPT', './checkpoints/dpvo.pth')

cfg.FEATURES_EXTR_CKPT = get_env_var('FEATURES_EXTR_CKPT', './checkpoints/hmr2a.ckpt')
