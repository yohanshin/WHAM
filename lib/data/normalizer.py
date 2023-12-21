import torch
import random
import numpy as np

from lib.utils.imutils import transform_keypoints

class Normalizer:
    def __init__(self, cfg):
        pass
        
    def __call__(self, kp_2d, res, cam_intrinsics, patch_width=224, patch_height=224, bbox=None, mask=None):
        if bbox is None:
            bbox = compute_bbox_from_keypoints(kp_2d, do_augment=True, mask=mask)
        
        out_kp_2d = self.bbox_normalization(kp_2d, bbox, res, patch_width, patch_height)
        return out_kp_2d, bbox
        
    def bbox_normalization(self, kp_2d, bbox, res, patch_width, patch_height):
        to_torch = False
        if isinstance(kp_2d, torch.Tensor):
            to_torch = True
            kp_2d = kp_2d.numpy()
            bbox = bbox.numpy()
        
        out_kp_2d = np.zeros_like(kp_2d)
        for idx in range(len(out_kp_2d)):
            out_kp_2d[idx] = transform_keypoints(kp_2d[idx], bbox[idx][:3], patch_width, patch_height)[0]
            out_kp_2d[idx] = normalize_keypoints_to_patch(out_kp_2d[idx], patch_width)
        
        if to_torch:
            out_kp_2d = torch.from_numpy(out_kp_2d)
            bbox = torch.from_numpy(bbox)
        
        centers = normalize_keypoints_to_image(bbox[:, :2].unsqueeze(1), res).squeeze(1)
        scale = bbox[:, 2:] * 200 / res.max()
        location = torch.cat((centers, scale), dim=-1)
        
        out_kp_2d = out_kp_2d.reshape(out_kp_2d.shape[0], -1)
        out_kp_2d = torch.cat((out_kp_2d, location), dim=-1)
        return out_kp_2d
        
        
def normalize_keypoints_to_patch(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d


def normalize_keypoints_to_image(x, res):
    res = res.to(x.device)
    scale = res.max(-1)[0].reshape(-1)
    mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1).to(x.device)
    x = (2 * x / scale.reshape(*[1 for i in range(len(x.shape[1:]))]) - \
        mean.reshape(*[1 for i in range(len(x.shape[1:-1]))], -1))
    return x


def compute_bbox_from_keypoints(X, do_augment=False, mask=None):
    def smooth_bbox(bb):
        # Smooth bounding box detection
        import scipy.signal as signal
        smoothed = np.array([signal.medfilt(param, int(30 / 2)) for param in bb])
        return smoothed
    
    def do_augmentation(scale_factor=0.3, trans_factor=0.25):
        # _scaleFactor = random.uniform(1.0 - scale_factor, 1.2 + scale_factor)
        _scaleFactor = random.uniform(1.2 - scale_factor, 1.2 + scale_factor)
        _trans_x = random.uniform(-trans_factor, trans_factor)
        _trans_y = random.uniform(-trans_factor, trans_factor)
        
        return _scaleFactor, _trans_x, _trans_y
    
    if do_augment:
        scaleFactor, trans_x, trans_y = do_augmentation()
    else:
        scaleFactor, trans_x, trans_y = 1.2, 0.0, 0.0
    
    if mask is None:
        bbox = [X[:, :, 0].min(-1)[0], X[:, :, 1].min(-1)[0],
                X[:, :, 0].max(-1)[0], X[:, :, 1].max(-1)[0]]
    else:
        bbox = []
        for x, _mask in zip(X, mask):
            if _mask.sum() > 10: 
                _mask[:] = False
            _bbox = [x[~_mask, 0].min(-1)[0], x[~_mask, 1].min(-1)[0],
                    x[~_mask, 0].max(-1)[0], x[~_mask, 1].max(-1)[0]]
            bbox.append(_bbox)
        bbox = torch.tensor(bbox).T
    
    cx, cy = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = torch.stack((bbox_w, bbox_h)).max(0)[0]
    scale = bbox_size * scaleFactor
    bbox = torch.stack((cx + trans_x * scale, cy + trans_y * scale, scale / 200))
    if mask is not None:
        bbox = torch.from_numpy(smooth_bbox(bbox.numpy()))
    
    return bbox.T