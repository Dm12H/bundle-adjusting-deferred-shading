import re
import sys
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
    AABB, read_views, read_mesh, write_mesh, visualize_views, generate_mesh, mesh_generator_names, get_pose_init, quat_to_rot
)

from evaluation.vis import vis_cameras, draw_pcd, pcl_chamfer_color
from evaluation.metrics import (
    prepare_pcl, chamfer_dist, psnr_metric, camera_est_errors
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

        self.views = self.load_views()
        self.bbox = self.load_bbox(self.paths.input_bbox,
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

        self._normalize_views()
        self.mesh = self.mesh_initial
        self.shader_optimizer = torch.optim.Adam(self.shader.parameters(), lr=self.shader_lr)
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)
        if self.params.train_pose:
            self.pose_vecs = torch.nn.Embedding(len(self.views), 7).to(device)
            self.pose_vecs.weight.data.copy_(get_pose_init(self.views, self.device))
            self.pose_optimizer= torch.optim.Adam(self.pose_vecs.parameters(), lr=self.pose_lr)
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
        if cfg.run.run_name is not None:
            run_name = cfg.run.run_name
        else:
            run_name = cfg.paths.input_dir.parent.name

        experiment_dir = cfg.paths.output_dir / run_name
        self.out_dir = experiment_dir
        self.id = int(re.match(r"\d+", run_name).group())

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
                self.params.gt_masks / f"ObsMask{self.id}_10.mat")
            self.gt_ground = loadmat(
                self.params.gt_masks / f"Plane{self.id}.mat")['P']
        else:
            self.gt_masks = None
            self.gt_ground = None

        if self.paths.gt_points is not None:
            points_path = self.params.gt_points / f"stl{self.id:03d}_total.ply"
            self.gt_points = o3d.io.read_point_cloud(
                points_path.absolute().as_posix()
            )
        else:
            self.gt_points = None

    def load_views(self):
        # Read the views
        views = read_views(
            self.paths.input_dir,
            scale=self.run_params.image_scale,
            device=self.device,
            approx=self.params.train_pose)
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
            mesh = read_mesh(args.initial_mesh_name, device=device)
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

        # record camera pose errors
        dir_errors, pos_errors = [], []
        for view in self.views:
            angle_err, pos_err = camera_est_errors(view)
            dir_errors.append(angle_err)
            pos_errors.append(pos_err)
        self.summary.add_scalar("Cameras/direction_err", np.mean(dir_errors), self.iteration)
        self.summary.add_scalar("Cameras/position_err", np.mean(pos_errors), self.iteration)
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
            [view],
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
        with (torch.no_grad()):
            use_fixed_views = len(self.run_params.visualization_views) > 0
            if use_fixed_views:
                view_indices = self.run_params.visualization_views
            else:
                view_indices=[np.random.choice(list(range(len(self.views))))]
            vi_psnr = []
            for vi in view_indices:
                debug_view = self.views[vi]
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
                vis_cameras(self.views,  cam_path)

            self.summary.add_scalar("Errors/PSNR", np.mean(vi_psnr), self.iteration)



    def save_mesh(self):
        with torch.no_grad():
            mesh_for_writing = self.space_normalization.denormalize_mesh(
                self.mesh.detach().to('cpu'))
            write_mesh(self.meshes_save_path / f"mesh_{self.iteration:06d}.obj",
                       mesh_for_writing)
        self.shader.save(self.shaders_save_path / f'shader_{self.iteration:06d}.pt')

    def run_to_completion(self, total=None):
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
        if self.gt_points is not None and self.gt_masks is not None:
            self.denormalize()
            denorm_mesh = self.space_normalization.denormalize_mesh(
                self.mesh.detach().to('cpu')
            )
            gt_cloud, eval_cloud = prepare_pcl(
                denorm_mesh, self.gt_points, self.gt_masks, self.gt_ground)
            colored_cloud = pcl_chamfer_color(gt_cloud, eval_cloud)
            # view 31 has a good viewing angle with minor correction
            full_out_path = self.out_dir / "chamfer_vis.png"
            draw_pcd(
                colored_cloud,
                self.views[32],
                full_out_path.absolute().as_posix(),
                y_correction=5,
                scale=self.run_params.image_scale)


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

if __name__ == '__main__':
    parser = ArgumentParser(description='Multi-View Mesh Reconstruction with Neural Deferred Shading', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=Path, default="./data", help="Path to the input data")
    parser.add_argument('--input_bbox', type=Path, default=None, help="Path to the input bounding box. If None, it is computed from the input mesh")
    parser.add_argument('--train_pose', action="store_true", help="whether to train camera poses")
    parser.add_argument('--output_dir', type=Path, default="./out", help="Path to the output directory")
    parser.add_argument('--gt_points', type=Path, default=None, help="Path to the ground truth  points directory")
    parser.add_argument('--gt_masks', type=Path, default=None, help="Path to the ground truth masks")
    parser.add_argument('--initial_mesh', type=str, default="vh32", help="Initial mesh, either a path or one of [vh16, vh32, vh64, sphere16]")
    parser.add_argument('--image_scale', type=int, default=1, help="Scale applied to the input images. The factor is 1/image_scale, so image_scale=2 halves the image size")
    parser.add_argument("--timeout", type=int, default=1200, help="timeout for 1000 iterations to restart")
    parser.add_argument('--iterations', type=int, default=2000, help="Total number of iterations")
    parser.add_argument('--run_name', type=str, default=None, help="Name of this run")
    parser.add_argument('--lr_vertices', type=float, default=1e-3, help="Step size/learning rate for the vertex positions")
    parser.add_argument('--lr_shader', type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    parser.add_argument('--lr_pose', type=float, default=1e-3, help="Step size/learning rate for learning camera poses")
    parser.add_argument('--upsample_iterations', type=int, nargs='+', default=[500, 1000, 1500], help="Iterations at which to perform mesh upsampling")
    parser.add_argument('--rebuild_iterations', type=int, nargs='+',default=[500, 1000, 1500], help="Iterations at which to perform mesh upsampling")
    parser.add_argument('--save_frequency', type=int, default=100, help="Frequency of mesh and shader saving (in iterations)")
    parser.add_argument('--visualization_frequency', type=int, default=100, help="Frequency of shader visualization (in iterations)")
    parser.add_argument('--visualization_views', type=int, nargs='+', default=[], help="Views to use for visualization. By default, a random view is selected each time")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")
    parser.add_argument('--weight_mask', type=float, default=2.0, help="Weight of the mask term")
    parser.add_argument('--weight_normal', type=float, default=0.1, help="Weight of the normal term")
    parser.add_argument('--weight_laplacian', type=float, default=40.0, help="Weight of the laplacian term")
    parser.add_argument('--weight_shading', type=float, default=1.0, help="Weight of the shading term")
    parser.add_argument('--shading_percentage', type=float, default=0.75, help="Percentage of valid pixels considered in the shading loss (0-1)")
    parser.add_argument('--hidden_features_layers', type=int, default=3, help="Number of hidden layers in the positional feature part of the neural shader")
    parser.add_argument('--hidden_features_size', type=int, default=256, help="Width of the hidden layers in the neural shader")
    parser.add_argument('--fourier_features', type=str, default='positional', choices=(['none', 'gfft', 'positional']), help="Input encoding used in the neural shader")
    parser.add_argument('--activation', type=str, default='relu', choices=(['relu', 'sine']), help="Activation function used in the neural shader")
    parser.add_argument('--fft_scale', type=int, default=4, help="Scale parameter of frequency-based input encodings in the neural shader")

    # Add module arguments
    ViewSampler.add_arguments(parser)

    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")
    mp.set_start_method("spawn")
    recover=True
    reconstruct(args, device)
    # while True:
    #     try:
    #         print("running job!")
    #         training_job = mp.Process(target=reconstruct, args=(args, device))
    #         training_job.daemon = True
    #         training_job.start()
    #         print(f"timer set to {args.timeout // args.image_scale}")
    #         training_job.join(timeout=(args.timeout // args.image_scale))
    #         if training_job.is_alive():
    #             print("timeout's reached!, rerunning!")
    #             training_job.terminate()
    #         break
    #     except KeyboardInterrupt:
    #         sys.exit()





