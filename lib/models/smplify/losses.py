import torch

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)


class SMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 res,
                 cam_intrinsics,
                 init_pose, 
                 device,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.init_pose = torch.from_numpy(init_pose).float().to(device)
        
    def forward(self, output, params, input_keypoints, bbox, 
                reprojection_weight=100., regularize_weight=60.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=20.0, sigma=100):
        
        pose, shape, cam = params
        scale = bbox[..., 2:].unsqueeze(-1) * 200.
        
        # Loss 1. Data term
        pred_keypoints = output.full_joints2d[..., :17, :]
        joints_conf = input_keypoints[..., -1:]
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf) / scale).mean()
        
        # Loss 2. Regularization term
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean()
        
        # Loss 3. Shape prior and consistency error
        consistency_error = shape.std(dim=1).mean()
        sprior_error = torch.linalg.norm(shape, dim=-1).mean()
        shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        # Loss 4. Smooth loss
        pose_diff = compute_jitter(pose).mean()
        cam_diff = compute_jitter(cam).mean()
        smooth_error = pose_diff + cam_diff
        
        # Sum up losses
        loss = {
            'reprojection': reprojection_weight * reprojection_error,
            'regularize': regularize_weight * regularize_error,
            'shape': shape_error,
            'smooth': smooth_weight * smooth_error
        }
        
        return loss
        
    def create_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints):
        
        def closure():
            optimizer.zero_grad()
            output = smpl(*params, cam_intrinsics=self.cam_intrinsics, bbox=bbox, res=self.res)
            
            loss_dict = self.forward(output, params, input_keypoints, bbox)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure