import numpy as np
import open3d as o3d
from skimage.metrics import structural_similarity

from nds.utils.geometry import mesh_to_pcl, downsample_cloud, get_rigid_transform


def psnr_metric(view, rec_im):
    """
    Evaluates average PSNR for all views of the model
    Args:
        view: view representing images taken from multiple cameras
        model: reconstruction model with trained shader and mesh

    Returns: float

    """

    gt_im = view.color.cpu().numpy().astype(np.float64)
    mask = view.mask
    rec_im = rec_im.cpu().numpy().astype(np.float64)
    mask = mask.cpu().numpy() > 0
    n_pix = np.sum(mask).astype(np.float64)

    #MSE
    mse = np.sum((mask * (gt_im - rec_im) ** 2)) / n_pix
    if mse == 0:
        #identical images
        return 100

    psnr = (20 * np.log10(1. / np.sqrt(mse)))
    return psnr


def ssim_metric(view, rec_im):
    """
        Evaluates average SSIM for all views of the model
    Args:
        view: a view representing images taken from multiple cameras
        model: reconstruction model with trained shader and mesh

    Returns: float

    """

    gt_im = view.color.cpu().numpy().astype(np.float64)
    mask = view.mask.cpu().numpy().astype(np.float64)
    rec_im = rec_im.cpu().numpy().astype(np.float64)
    gt_masked = gt_im * (mask > 0).astype(np.float64)
    rec_masked = rec_im * (mask > 0).astype(np.float64)
    *_, ssim_full = structural_similarity(
        gt_masked,
        rec_masked,
        full=True,
        data_range=1.0,
        channel_axis=2
    )
    ssim_full_masked = ssim_full * (mask > 0)
    ssimm_per_channel = np.mean(ssim_full_masked, axis=2)
    ssim_mean = np.sum(ssimm_per_channel) / np.sum(mask)
    return ssim_mean


def prepare_pcl(mesh, gt_cloud, mask_mat, ground_plane):
    """
    turns reonstructed mesh into point cloud and prepared DTU-format ground
    truth point cloud for evaluation
    Args:
        mesh: reconstructed mesh from the model
        gt_cloud: ground truth point cloud for evaluation
        mask_mat: .mat file storing bounding box, voxel mask
            and scaling factor for it
        ground_plane: vector of length {4} storing coeffs for ground plane in
            point cloud

    Returns:

    """
    obs_mask = mask_mat["ObsMask"]
    bb = mask_mat["BB"]
    res = mask_mat["Res"]

    mesh_points = mesh_to_pcl(mesh)
    eval_points = downsample_cloud(mesh_points)

    gt_points = np.asarray(gt_cloud.points)
    indices = np.around((gt_points - bb[:1]) / res).astype(np.int32)
    point_farther_then_lcorn = np.all(indices >= 0, axis=1)
    point_closer_then_rcorn = np.all(indices < [obs_mask.shape], axis=1)
    points_inside_bbox = np.logical_and(point_farther_then_lcorn,
                                        point_closer_then_rcorn)

    pcloud_mask = np.array(
        [bool(obs_mask[tuple(idx)]) if points_inside_bbox[i] else False
         for i, idx in enumerate(indices)]
    )

    points_hom = np.concatenate([gt_points, np.ones_like(gt_points[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * points_hom).sum(-1) > 0
    full_mask = np.logical_and(pcloud_mask, above)

    p_index = np.arange(len(full_mask))
    selected_indices = p_index[full_mask]
    masked_cloud = gt_cloud.select_by_index(selected_indices)

    eval_cloud = o3d.geometry.PointCloud()
    eval_cloud.points = o3d.pybind.utility.Vector3dVector(eval_points)

    return masked_cloud, eval_cloud


def chamfer_dist(gt_cloud, eval_cloud, max_dist=10):
    mesh_to_pcl_dist = np.asarray(gt_cloud.compute_point_cloud_distance(eval_cloud))
    within_max_dist = mesh_to_pcl_dist < max_dist
    mean_dist = np.mean(mesh_to_pcl_dist[within_max_dist])
    return mean_dist


def camera_est_errors(view, normalizer, rigid):

    camera = view.camera
    K_cur, R_cur, t_cur = view.transform(normalizer.A_inv, normalizer.A)
    R_rigid, t_rigid, c = rigid

    # calculate degree disparity
    view_direction = np.squeeze(camera.look_dir(R_cur))
    view_direction = view_direction / np.linalg.norm(view_direction)
    view_direction = R_rigid @ view_direction

    gt_view_direction = np.squeeze(camera.look_dir(camera.R_gt))
    gt_view_direction = gt_view_direction / np.linalg.norm(gt_view_direction)

    dot = np.clip(view_direction.dot(gt_view_direction), a_min=0, a_max=1)
    angle_rad = np.abs(np.arccos(dot))
    angle_deg = 180 * angle_rad / np.pi

    # calculate distance disparity
    camera_pos = camera.center_general(R_cur, t_cur)
    camera_pos = c * R_rigid @ camera_pos + t_rigid
    camera_pos_gt = camera.center_general(camera.R_gt, camera.t_gt)
    dist = np.sqrt(np.sum((camera_pos - camera_pos_gt) ** 2))
    return angle_deg, dist


def mean_cam_est_err(views, normalizer):
    # record camera pose errors
    dir_errors, pos_errors = [], []
    rigid_t = get_rigid_transform(views)
    for view in views:
        angle_err, pos_err = camera_est_errors(view, normalizer, rigid_t)
        dir_errors.append(angle_err)
        pos_errors.append(pos_err)
    mdir_error = np.mean(dir_errors)
    mpos_error = np.mean(pos_errors)
    return mdir_error, mpos_error
