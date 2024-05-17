import numpy as np
import torch

class Camera:
    """ Camera in OpenCV format.
        
    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """

    def __init__(self, K, R, t, device='cpu'):
        self.K = K.to(device) if torch.is_tensor(K) else torch.FloatTensor(K).to(device)
        self.R = R.to(device) if torch.is_tensor(R) else torch.FloatTensor(R).to(device)
        self.R_gt = R
        self.t = t.to(device) if torch.is_tensor(t) else torch.FloatTensor(t).to(device)
        self.t_gt = t
        self.device = device

    def to(self, device="cpu"):
        self.K = self.K.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.device = device
        return self

    @property
    def center(self):
        return -self.R.t() @ self.t

    @property
    def center_gt(self):
        return -self.R_gt.T @ self.t_gt

    @property
    def P(self):
        return self.K @ self.Rt
    @property
    def Ex(self):
        rt_mat = self.Rt
        hom_vec = torch.tensor(
            [[0, 0, 0, 1]],
            dtype=rt_mat.dtype,
            device=rt_mat.device)
        return torch.cat([rt_mat, hom_vec], dim=0)

    @property
    def Rt(self):
        return torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)

    @property
    def true_look_dir(self):
        with torch.no_grad():

            view_dir_cam = np.array([0, 0, -1],
                                        dtype=np.float64)
            return self.R_gt.dot(view_dir_cam)

    @property
    def look_dir(self):
        with torch.no_grad():
            view_dir_cam = torch.tensor([0, 0, -1],
                                        dtype=torch.float).to(self.device)
            vec = self.R @ (torch.unsqueeze(view_dir_cam, dim=1))
            return torch.squeeze(vec)

    @property
    def intinsics(self):
        intrinsic_mat = self.K.cpu().numpy()
        fx = intrinsic_mat[0,0]
        fy = intrinsic_mat[1,1]
        cx = intrinsic_mat[0, 2]
        cy = intrinsic_mat[1, 2]
        return fx, fy, cx, cy