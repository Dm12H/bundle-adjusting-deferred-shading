import numpy as np
import cv2
import torch
from torch.nn import functional as F

def decompose(P):
    K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    c = c[:3, 0] / c[3]
    t = - R @ c
    # ensure unique scaling of K matrix
    K = K / K[2,2]
    return K, R, t

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def quat_to_rot(q, device):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).to(device)
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R, device):
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4)).to(device)

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1] = (R21-R12)/(4*q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_pose_init(views, device):
    init_pose = []
    for view in views:
        pose = view.camera.Rt
        init_pose.append(pose)
    init_pose = torch.cat(
        [torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose],
        0).to(device)
    init_quat = rot_to_quat(init_pose[:, :3, :3], device)
    init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

    return init_quat
