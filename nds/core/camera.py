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

    @staticmethod
    def center_general(R, t):
        return -R.T @ t

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

    @staticmethod
    def look_dir(R):
        with torch.no_grad():

            view_dir_cam = np.array([0, 0, -1],
                                        dtype=np.float64)
            return R.dot(view_dir_cam)

    @property
    def intinsics(self):
        intrinsic_mat = self.K.cpu().numpy()
        fx = intrinsic_mat[0,0]
        fy = intrinsic_mat[1,1]
        cx = intrinsic_mat[0, 2]
        cy = intrinsic_mat[1, 2]
        return fx, fy, cx, cy

    def perturb_lookat(self, err: float):

        # sample a new lookat direction
        err_angle = err * np.pi / 180
        cone_r = np.tan(err_angle)
        r = cone_r * np.sqrt(np.random.random())
        angle = np.random.random() * 2 * np.pi
        u = r * np.cos(angle)
        v = r * np.sin(angle)
        new_lookat = np.array([u, v, -1], dtype=np.float32)
        unit_lookat = new_lookat / np.linalg.norm(new_lookat)

        # get rotation matrix from 2 vectors
        old_lookat = np.array([0, 0, -1], dtype=np.float32)
        cross = np.cross(old_lookat, unit_lookat)
        cos = np.dot(old_lookat, unit_lookat)
        U = np.array([[0, -cross[2], cross[1]],
                      [cross[2], 0, -cross[0]],
                      [-cross[1], cross[0], 0]])
        Usq = U.dot(U)
        R = np.identity(3) + U + (1 / 1 + cos) * Usq

        # apply new rotation angle
        R_tens = torch.from_numpy(R).to(torch.float).to(self.device)
        self.R = R_tens.t() @ self.R

    def perturb_position(self, err,  bbox_center):
        cam_center = self.center.cpu().numpy()
        dist = np.sqrt(np.sum((cam_center - bbox_center) ** 2))
        shift_vec = np.random.normal(size=3)
        unit_shift = shift_vec / np.linalg.norm(shift_vec)
        percent_err = err / 100
        final_shift = dist * percent_err * unit_shift

        shift_t = torch.from_numpy(final_shift).to(torch.float).to(self.device)
        self.t = shift_t + self.t




