from copy import deepcopy

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

def get_camera_mesh(scale=1):
    vertices = np.array([[-0.5, -0.5, 1],
                         [0.5, -0.5, 1],
                         [0.5,0.5,1],
                         [-0.5,0.5,1],
                         [0,0,0]]) * scale
    wireframe = vertices[[0,1,2,3,0,4,1,2,4,3]]
    return wireframe


def vis_cameras(views, normalizer, out_path, colors=("blue", "magenta")):
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": "3d"})
    cam_mesh = get_camera_mesh(scale=50)
    color_gt, color_est = colors
    for view in views:
        cam = view.camera
        _, R, t = view.transform(normalizer.A_inv, normalizer.A)
        R_gt = cam.R_gt
        t_gt = cam.t_gt

        cam_gt = R_gt.T @ (cam_mesh - t_gt).T
        cam_est = R.T @ (cam_mesh - t).T

        x_gt, z_gt, y_gt = np.split(cam_gt,3, axis=0)
        x, z, y = np.split(cam_est, 3, axis=0)

        dist = np.sqrt(np.sum((cam_gt - cam_est) ** 2, axis=0))
        ax.plot_wireframe(x_gt, y_gt, z_gt, color=color_gt)
        if np.mean(dist) < 1:
            continue
        ax.plot_wireframe(x, y, z, color=color_est)


    ax.set(xlabel="X",
           ylabel="Z",
           zlabel="Y")
    fig.savefig(out_path)
    matplotlib.pyplot.close(fig)


def pcl_chamfer_color(gt_pcl, mesh_pcl, clip=10):
    mesh_to_pcl_dist = np.asarray(
        mesh_pcl.compute_point_cloud_distance(gt_pcl)
    )
    clipped_distances = np.clip(mesh_to_pcl_dist, a_min=0, a_max=clip) / clip
    cmap = plt.get_cmap('jet')
    colors = cmap(clipped_distances)[:, :3]
    vis_mesh = deepcopy(mesh_pcl)
    vis_mesh.colors = o3d.pybind.utility.Vector3dVector(colors)
    return vis_mesh


def draw_pcd(pcd, view, out_path, y_correction=0, scale=1):
    vis = o3d.visualization.Visualizer()
    camera = view.camera
    cam_params = o3d.camera.PinholeCameraParameters()
    extrinsic = camera.Ex.cpu().numpy().astype(np.float64)
    thet = y_correction * np.pi / 180
    corr_mat = np.array(
        [[1, 0, 0],
         [0, np.cos(thet), -np.sin(thet)],
         [0, np.sin(thet), np.cos(thet)]]
    )
    rot_mat = extrinsic[:3,:3]
    final_rot_mat = corr_mat @ rot_mat
    extrinsic[:3, :3] = final_rot_mat
    h, w, _ = view.color.shape
    w *= scale
    h *= scale
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    fx, fy, cx, cy = camera.intinsics
    fx *= scale
    fy *= scale
    intrinsic.set_intrinsics(w, h, fx, fy, w/2, h/2)
    cam_params.intrinsic = intrinsic
    cam_params.extrinsic = extrinsic
    vis.create_window(width=w, height=h)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params, True)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    vis.destroy_window()
