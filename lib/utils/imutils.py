import cv2
import torch
import random
import numpy as np
from . import transforms

def do_augmentation(scale_factor=0.2, trans_factor=0.1):
    scale = random.uniform(1.2 - scale_factor, 1.2 + scale_factor)
    trans_x = random.uniform(-trans_factor, trans_factor)
    trans_y = random.uniform(-trans_factor, trans_factor)
    
    return scale, trans_x, trans_y

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def crop_cliff(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br


def obtain_bbox(center, scale, res, org_res):
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    # Range to sample from original image
    old_x = max(0, ul[0]), min(org_res[0], br[0])
    old_y = max(0, ul[1]), min(org_res[1], br[1])
    
    return old_x, old_y


def cam_crop2full(crop_cam, bbox, full_img_shape, focal_length=None):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    
    cx = bbox[..., 0].clone(); cy = bbox[..., 1].clone(); b = bbox[..., 2].clone() * 200
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, :, 0] + 1e-9
    
    if focal_length is None:
        focal_length = (img_w * img_w + img_h * img_h) ** 0.5
    
    tz = 2 * focal_length.unsqueeze(-1) / bs
    tx = (2 * (cx - w_2.unsqueeze(-1)) / bs) + crop_cam[:, :, 1]
    ty = (2 * (cy - h_2.unsqueeze(-1)) / bs) + crop_cam[:, :, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def cam_pred2full(crop_cam, center, scale, full_img_shape, focal_length=2000.,):
    """
    Reference CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation
    
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, ) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    
    # img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    img_w, img_h = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def cam_full2pred(full_cam, center, scale, full_img_shape, focal_length=2000.):
    # img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    img_w, img_h = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    
    bs = (2 * focal_length / full_cam[:, 2])
    _s = bs / b
    _tx = full_cam[:, 0] - (2 * (cx - w_2) / bs)
    _ty = full_cam[:, 1] - (2 * (cy - h_2) / bs)
    crop_cam = torch.stack([_s, _tx, _ty], dim=-1)
    return crop_cam


def obtain_camera_intrinsics(image_shape, focal_length):
    res_w = image_shape[..., 0].clone()
    res_h = image_shape[..., 1].clone()
    K = torch.eye(3).unsqueeze(0).expand(focal_length.shape[0], -1, -1).to(focal_length.device)
    K[..., 0, 0] = focal_length.clone()
    K[..., 1, 1] = focal_length.clone()
    K[..., 0, 2] = res_w / 2
    K[..., 1, 2] = res_h / 2
    
    return K.unsqueeze(1)


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def transform_keypoints(kp_2d, bbox, patch_width, patch_height):

    center_x, center_y, scale = bbox[:3]
    width = height = scale * 200
    # scale, rot = 1.2, 0
    scale, rot = 1.0, 0

    # generate transformation
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale,
        rot,
        inv=False,
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return kp_2d, trans


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def compute_cam_intrinsics(res):
    img_w, img_h = res
    focal_length = (img_w * img_w + img_h * img_h) ** 0.5
    cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
    cam_intrinsics[:, 0, 0] = focal_length
    cam_intrinsics[:, 1, 1] = focal_length
    cam_intrinsics[:, 0, 2] = img_w/2.
    cam_intrinsics[:, 1, 2] = img_h/2.
    return cam_intrinsics


def flip_kp(kp, img_w=None):
    """Flip keypoints."""
    
    flipped_parts = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    kp = kp[..., flipped_parts, :]
    
    if img_w is not None:
        # Assume 2D keypoints
        kp[...,0] = img_w - kp[...,0]
    return kp


def flip_bbox(bbox, img_w, img_h):
    center = bbox[..., :2]
    scale = bbox[..., -1:]
    
    WH = np.ones_like(center)
    WH[..., 0] *= img_w
    WH[..., 1] *= img_h
    
    center = center - WH/2
    center[...,0] = - center[...,0]
    center = center + WH/2
    
    flipped_bbox = np.concatenate((center, scale), axis=-1)
    return flipped_bbox


def flip_pose(rotation, representation='rotation_6d'):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    
    BN = rotation.shape[0]
    
    if representation == 'axis_angle':
        pose = rotation.reshape(BN, -1).transpose(0, 1)
    elif representation == 'matrix':
        pose = transforms.matrix_to_axis_angle(rotation).reshape(BN, -1).transpose(0, 1)
    elif representation == 'rotation_6d':
        pose = transforms.matrix_to_axis_angle(
            transforms.rotation_6d_to_matrix(rotation)
        ).reshape(BN, -1).transpose(0, 1)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    SMPL_POSE_FLIP_PERM = []
    for i in SMPL_JOINTS_FLIP_PERM:
        SMPL_POSE_FLIP_PERM.append(3*i)
        SMPL_POSE_FLIP_PERM.append(3*i+1)
        SMPL_POSE_FLIP_PERM.append(3*i+2)
    
    pose = pose[SMPL_POSE_FLIP_PERM]
    
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    pose = pose.transpose(0, 1).reshape(BN, -1, 3)
    
    if representation == 'aa':
        return pose
    elif representation == 'rotmat':
        return transforms.axis_angle_to_matrix(pose)
    else:
        return transforms.matrix_to_rotation_6d(
            transforms.axis_angle_to_matrix(pose)
        )
        
def avg_preds(rotation, shape, flipped_rotation, flipped_shape, representation='rotation_6d'):
    # Rotation
    flipped_rotation = flip_pose(flipped_rotation, representation=representation)
    
    if representation != 'matrix':
        flipped_rotation = eval(f'transforms.{representation}_to_matrix')(flipped_rotation)
        rotation = eval(f'transforms.{representation}_to_matrix')(rotation)
    
    avg_rotation = torch.stack([rotation, flipped_rotation])
    avg_rotation = transforms.avg_rot(avg_rotation)
    
    if representation != 'matrix':
        avg_rotation = eval(f'transforms.matrix_to_{representation}')(avg_rotation)
    
    # Shape
    avg_shape = (shape + flipped_shape) / 2.0
    
    return avg_rotation, avg_shape