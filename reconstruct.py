import re
import sys
import time
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import meshzoo
import numpy as np
import open3d as o3d
from pathlib import Path

from scipy.io import loadmat
from pyremesh import remesh_botsch
import multiprocessing as mp
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nds.core import (
    Mesh, Renderer
)
from nds.losses import (
    mask_loss, normal_consistency_loss, laplacian_loss, shading_loss
)
from nds.modules import (
    SpaceNormalization, NeuralShader, ViewSampler
)
from nds.utils import (
    AABB, read_views, read_mesh, write_mesh, visualize_views,
    generate_mesh, mesh_generator_names, get_pose_init, quat_to_rot,
    get_camps_rigid_transform, create_t_from_rigid
)

from evaluation.vis import vis_cameras
from evaluation.metrics import (
    prepare_pcl, chamfer_dist, psnr_metric, mean_cam_est_err, ssim_metric
)


class Reconstructor:
    def __init__(self,
                 cfg,
                 device,
                 iter_n=1):

        self.iteration=iter_n
        self.paths = cfg.paths
        self.params = cfg.params
        self.run_params = cfg.run
        self.device = device

        if cfg.run.run_name is not None:
            subdir = cfg.run.run_name
            self.run_name = subdir
        else:
            raise ValueError("run_name is expected")

        if cfg.run.subdir_name is not None:
            subdir = cfg.run.subdir_name

        self.views = self.load_views()
        self.bbox = self.load_bbox(
            self.paths.input_bbox / self.run_name / "bbox.txt",
            self.params.initial_mesh)
        self.mesh_initial = self.load_mesh(self.views, self.bbox)
        # Create the optimizer for the neural shader
        self.shader = NeuralShader(
            hidden_features_layers=self.params.shader.hidden_features_layers,
            hidden_features_size=self.params.shader.hidden_features_size,
            fourier_features=self.params.shader.fourier_features,
            activation=self.params.shader.activation,
            fft_scale=self.params.shader.fft_scale,
            last_activation=torch.nn.Sigmoid,
            device=device)
        self.vertices_lr = self.params.lrs.lr_vertices
        self.shader_lr = self.params.lrs.lr_shader
        self.pose_lr = self.params.lrs.lr_pose
        self.loss_weights = {
            "mask": self.params.loss_weights.weight_mask,
            "normal": self.params.loss_weights.weight_normal,
            "laplacian": self.params.loss_weights.weight_laplacian,
            "shading": self.params.loss_weights.weight_shading
        }
        self.space_normalization = SpaceNormalization(self.bbox.corners)

        if self.run_params.perturbs is not None:
            bbox_center = self.bbox.center
            for view in self.views:
                view.camera.perturb_position(
                    self.run_params.perturbs.pos, bbox_center
                )
                view.camera.perturb_lookat(
                    self.run_params.perturbs.dir
                )

        self._normalize_views()
        self.mesh = self.mesh_initial
        self.shader_optimizer = torch.optim.Adam(self.shader.parameters(), lr=self.shader_lr)
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)
        if self.params.train_pose:
            self.pose_vecs = torch.nn.Embedding(len(self.views), 7).to(device)
            self.pose_vecs.weight.data.copy_(get_pose_init(self.views, self.device))
            self.pose_optimizer = torch.optim.Adam(self.pose_vecs.parameters(), lr=self.pose_lr)
        else:
            self.pose_vecs = None
            self.pose_optimizer = None

        # Configure the renderer
        self.renderer = Renderer(device=device)
        self.renderer.set_near_far(self.views, torch.from_numpy(self.bbox.corners).to(device), epsilon=0.5)

        # Configure the view sampler
        self.view_sampler = ViewSampler(
            views=self.views,
            **ViewSampler.get_parameters(cfg.params.view_sampler))

        # Set up save paths

        experiment_dir = cfg.paths.output_dir / subdir
        self.exp_dir = experiment_dir
        self.id = int(re.match(r"\d+", self.run_name).group())

        self.images_save_path = experiment_dir / "images"
        self.meshes_save_path = experiment_dir / "meshes"
        self.shaders_save_path = experiment_dir / "shaders"
        self.cameras_save_path = experiment_dir / "cameras"
        self._create_dirs()
        self.summary = SummaryWriter(experiment_dir)
        #bundle optimizers
        self.optimizers = [self.vertices_optimizer, self.shader_optimizer]
        if self.params.train_pose:
            self.optimizers += [self.pose_optimizer]

        if self.paths.gt_masks is not None:
            self.gt_masks = loadmat(
                self.paths.gt_masks / f"ObsMask{self.id}_10.mat")
            self.gt_ground = loadmat(
                self.paths.gt_masks / f"Plane{self.id}.mat")['P']
        else:
            self.gt_masks = None
            self.gt_ground = None

        if self.paths.gt_points is not None:
            points_path = self.paths.gt_points / f"stl/stl{self.id:03d}_total.ply"
            self.gt_points = o3d.io.read_point_cloud(
                points_path.absolute().as_posix()
            )
        else:
            self.gt_points = None

    def load_views(self):
        # Read the views
        views = read_views(
            self.paths.input_dir / self.run_name / "views",
            scale=self.run_params.image_scale,
            device=self.device)
        return views

    def load_mesh(self, views, bbox):
        # Obtain the initial mesh and compute its connectivity
        if self.params.initial_mesh in mesh_generator_names:
            # Use args.initial_mesh as mesh generator name
            if bbox is None:
                raise RuntimeError("Generated meshes require a bounding box.")
            mesh = generate_mesh(
                self.params.initial_mesh,
                views,
                bbox,
                device=self.device)
        else:
            # Use args.initial_mesh as path to the mesh
            mesh = read_mesh(self.params.initial_mesh, device=self.device)
        mesh.compute_connectivity()
        return mesh

    @staticmethod
    def load_bbox(input_bbox=None, mesh_initial=None):
        # Load the bounding box or create it from the mesh vertices
        if input_bbox is not None:
            aabb = AABB.load(input_bbox)
        elif mesh_initial is not None:
            aabb = AABB(mesh_initial.vertices.cpu().numpy())
        else:
            raise RuntimeError("must provide either mesh or bbox")
        return aabb

    def _create_dirs(self):
        self.images_save_path.mkdir(parents=True, exist_ok=True)
        self.meshes_save_path.mkdir(parents=True, exist_ok=True)
        self.shaders_save_path.mkdir(parents=True, exist_ok=True)
        if self.params.train_pose:
            self.cameras_save_path.mkdir(parents=True, exist_ok=True)

    def visualize_views(self, save_path):
        visualize_views(self.views, show=False, save_path=save_path)

    def _normalize_views(self):
        # Apply the normalizing affine transform, which maps the bounding box to
        # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
        normalizer = self.space_normalization
        self.views = normalizer.normalize_views(self.views)
        self.mesh_initial = normalizer.normalize_mesh(self.mesh_initial)
        self.bbox = normalizer.normalize_aabb(self.bbox)

    def reconstruction_step(self):

        mesh = self.mesh_initial.with_vertices(
            self.mesh_initial.vertices + self.vertex_offsets
        )

        # Sample a view subset
        views_subset, idx_subset = self.view_sampler(self.views)
        views_subset = [view.to(self.device) for view in views_subset]

        # Replace camera position with learnable params
        if self.params.train_pose:
            for view, idx in zip(views_subset, idx_subset):
                idx_on_device = torch.LongTensor([idx]).to(self.device)
                pose_embed = self.pose_vecs(idx_on_device)
                R_est = torch.squeeze(quat_to_rot(pose_embed[:, :4], self.device))
                t_est = torch.squeeze(pose_embed)[4:]
                view.camera.R = R_est
                view.camera.t = t_est

        # Render the mesh from the views
        # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
        gbuffers = self.renderer.render(
            views_subset, mesh,
            channels=['mask', 'position', 'normal'],
            with_antialiasing=True)

        loss = torch.tensor(0., device=self.device)

        # Combine losses and weights
        if (w := self.loss_weights['mask']) > 0:
            m_loss = mask_loss(views_subset, gbuffers)
            loss += m_loss * w
            self.summary.add_scalar("Loss/mask", m_loss, self.iteration)
        if (w := self.loss_weights['normal']) > 0:
            n_loss = normal_consistency_loss(mesh)
            loss += n_loss * w
            self.summary.add_scalar("Loss/normal", n_loss, self.iteration)
        if (w := self.loss_weights['laplacian']) > 0:
            l_loss = laplacian_loss(mesh)
            loss += l_loss * w
            self.summary.add_scalar("Loss/laplacian", l_loss, self.iteration)
        if (w := self.loss_weights['shading']) > 0:
            sh_loss = shading_loss(
                views_subset,
                gbuffers,
                shader=self.shader,
                shading_percentage=self.params.shader.shading_percentage
            )
            loss += sh_loss * w
            self.summary.add_scalar("Loss/shading", sh_loss, self.iteration)
        self.summary.add_scalar("Loss/total", loss, self.iteration)
        # Optimize
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        self.mesh = mesh
        dir_err, pos_err = mean_cam_est_err(self.views, self.space_normalization)

        self.summary.add_scalar("Cameras/direction_err", dir_err, self.iteration)
        self.summary.add_scalar("Cameras/position_err", pos_err, self.iteration)
        return loss.detach().cpu()

    def upsample_mesh(self):
        # Upsample the mesh by remeshing the surface with half the average edge length
        e0, e1 = self.mesh.edges.unbind(1)
        average_edge_length = torch.linalg.norm(
            self.mesh.vertices[e0] - self.mesh.vertices[e1], dim=-1).mean()
        v_upsampled, f_upsampled = remesh_botsch(
            self.mesh.vertices.cpu().detach().numpy().astype(np.float64),
            self.mesh.indices.cpu().numpy().astype(np.int32),
            h=float(average_edge_length / 2))
        v_upsampled = np.ascontiguousarray(v_upsampled)
        f_upsampled = np.ascontiguousarray(f_upsampled)

        self.mesh_initial = Mesh(v_upsampled, f_upsampled, device=self.device)
        self.mesh_initial.compute_connectivity()
        self.mesh = self.mesh_initial

        # Adjust weights and step size
        self.loss_weights['laplacian'] *= 4
        self.loss_weights['normal'] *= 4
        self.vertices_lr *= 0.75

        # Create a new optimizer for the vertex offsets
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)
        self.optimizers = [self.vertices_optimizer, self.shader_optimizer]
        if self.params.train_pose:
            self.optimizers += [self.pose_optimizer]

    def rebuild_mesh(self):
        self.mesh_initial = self.load_mesh(self.views, self.bbox)
        self.mesh_initial.compute_connectivity()
        self.mesh = self.mesh_initial

        # Create a new optimizer for the vertex offsets
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)

        self.optimizers = [self.vertices_optimizer, self.shader_optimizer, self.pose_optimizer]

    def render_view(self, view):
        gbuffer = self.renderer.render(
            [view.to(self.device)],
            self.mesh,
            channels=['mask', 'position', 'normal'],
            with_antialiasing=True)[0]
        position = gbuffer["position"]
        normal = gbuffer["normal"]
        view_direction = torch.nn.functional.normalize(
            view.camera.center - position, dim=-1)

        shaded_image = self.shader(position, normal, view_direction)
        mask = gbuffer["mask"]
        return shaded_image, mask, normal

    def visualize(self):
        with torch.no_grad():
            use_fixed_views = len(self.run_params.visualization_views) > 0
            if use_fixed_views:
                view_indices = self.run_params.visualization_views
            else:
                view_indices = [np.random.choice(list(range(len(self.views))))]
            vi_psnr = []
            for vi in view_indices:
                debug_view = self.views[vi].to(self.device)
                shaded_image, mask, normal = self.render_view(debug_view)
                # Save the shaded rendering
                shaded_image = shaded_image * mask + (1 - mask)
                shaded_path = (self.images_save_path / str(
                    vi) / "shaded") if use_fixed_views else (
                            self.images_save_path / "shaded")
                shaded_path.mkdir(parents=True, exist_ok=True)
                plt.imsave(shaded_path / f'neuralshading_{self.iteration}.png',
                           shaded_image.cpu().numpy())

                # Save a normal map in camera space
                normal_path = (self.images_save_path / str(
                    vi) / "normal") if use_fixed_views else (
                            self.images_save_path / "normal")
                normal_path.mkdir(parents=True, exist_ok=True)
                R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                 device=self.device, dtype=torch.float32)
                normal_image = (0.5 * (
                            normal @ debug_view.camera.R.T @ R.T + 1)) * \
                            mask + (1 - mask)
                plt.imsave(normal_path / f'neuralshading_{self.iteration}.png',
                           normal_image.cpu().numpy())

                # calc psnr
                vi_psnr.append(psnr_metric(debug_view, shaded_image))

            if self.params.train_pose:
                cam_path = self.cameras_save_path / f"cams_{self.iteration}.png"
                vis_cameras(self.views, self.space_normalization,  cam_path)

            self.summary.add_scalar("Errors/PSNR", np.mean(vi_psnr), self.iteration)

    def save_mesh(self):
        with torch.no_grad():
            mesh_for_writing = self.space_normalization.denormalize_mesh(
                self.mesh.detach().to('cpu'))
            write_mesh(self.meshes_save_path / f"mesh_{self.iteration:06d}.obj",
                       mesh_for_writing)
        self.shader.save(self.shaders_save_path / f'shader_{self.iteration:06d}.pt')

    def run_to_completion(self, total=None):
        start_time = time.time()
        if total is None:
            total = self.run_params.iterations
        progress_bar = tqdm(range(self.iteration, total + 1))
        for it in progress_bar:
            if it in self.run_params.upsample_iterations:
                self.upsample_mesh()
            if self.params.train_pose and it in self.run_params.rebuild_iterations:
                self.rebuild_mesh()
            step_loss = self.reconstruction_step()
            can_vis = self.run_params.visualization_frequency > 0 and self.shader is not None
            should_vis = it == 1 or it % self.run_params.visualization_frequency == 0
            if can_vis and should_vis:
                self.visualize()
            can_save = self.run_params.save_frequency > 0
            should_save = it == 1 or it % self.run_params.save_frequency == 0
            if can_save and should_save:
                self.save_mesh()
            self.iteration += 1
            progress_bar.set_postfix({'loss': step_loss})
            self.summary.flush()
        print("training finished")
        metrics = {}
        psnr_views = []
        ssim_views = []
        with torch.no_grad():
            for view in self.views:
                shaded_image, *_ = self.render_view(view.to(self.device))
                psnr_views.append(psnr_metric(view, shaded_image))
                ssim_views.append(ssim_metric(view, shaded_image))
        metrics["SSIM"] = np.mean(ssim_views)
        metrics["PSNR"] = np.mean(psnr_views)
        dir_err, pos_err = mean_cam_est_err(
            self.views,
            self.space_normalization)
        metrics["CAM_POS_ERR"] = pos_err
        metrics["CAM_DIR_ERR"] = dir_err
        end_time = time.time()
        metrics["TIME_ELAPSED"] = end_time - start_time
        if self.gt_points is not None and self.gt_masks is not None:
            self.denormalize()
            denorm_mesh = self.space_normalization.denormalize_mesh(
                self.mesh.detach().to('cpu')
            )
            gt_cloud, eval_cloud = prepare_pcl(
                denorm_mesh, self.gt_points, self.gt_masks, self.gt_ground)
            rigid = get_camps_rigid_transform(self.views)
            T_mat = create_t_from_rigid(*rigid)
            eval_cloud = eval_cloud.transform(T_mat)
            metrics["CHAMFER"] = chamfer_dist(gt_cloud, eval_cloud)
        # save cam params
        if self.params.train_pose:
            cams_out = self.exp_dir / "cam_params"
            cams_out.mkdir(parents=True, exist_ok=True)
            for view in self.views:
                view.save(cams_out)
        for path in (Path.cwd(), self.exp_dir):
            with open(path / 'metrics.json', 'w') as f:
                json.dump(metrics, f)

    def denormalize(self):
        self.space_normalization.denormalize_views(self.views)

    def recover(self):
        max_step = 0
        for fpath in self.meshes_save_path.iterdir():
            _, val = fpath.stem.split("_", maxsplit=1)
            max_step = max(max_step, int(val))
        mesh_path = self.meshes_save_path / f"mesh_{max_step:06d}.obj"
        shader_path = self.shaders_save_path / f"shader_{max_step:06d}.pt"
        self.iteration = max_step
        self.mesh = read_mesh(mesh_path, self.device)
        self.mesh_initial = self.space_normalization.normalize_mesh(self.mesh)
        self.mesh = self.mesh_initial
        self.shader = NeuralShader.load(shader_path, self.device)

        self.shader_optimizer = torch.optim.Adam(self.shader.parameters(), lr=self.shader_lr)
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)
        print(f"recovering at iteration {self.iteration}")


def reconstruct(args, device):
    reconstructor = Reconstructor(args, device)
    reconstructor.run_to_completion(args.iterations)
