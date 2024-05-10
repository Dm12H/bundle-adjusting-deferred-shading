import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import meshzoo
import numpy as np
from pathlib import Path
from pyremesh import remesh_botsch
from threading import Thread
import torch
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
    AABB, read_views, read_mesh, write_mesh, visualize_views, generate_mesh, mesh_generator_names
)

class Reconstructor:
    def __init__(self,
                 shader,
                 params,
                 device,
                 iter_n=1):

        self.iteration=iter_n
        self.views = self.load_views(params.input_dir, params.image_scale, params.pose_estimate)
        self.bbox = self.load_bbox(params.input_bbox, params.initial_mesh)
        self.mesh = self.load_mesh(params.initial_mesh, self.views, self.bbox)
        self.shader = shader
        self.vertices_lr = params.lr_vertices
        self.shader_lr = params.lr_shader
        self.params = params
        self.loss_weights = {
            "mask": params.weight_mask,
            "normal": params.weight_normal,
            "laplacian": params.weight_laplacian,
            "shading": params.weight_shading
        }
        self.space_normalization = SpaceNormalization(self.bbox.corners)
        self.device = device

        self._normalize_views()
        self.shader_optimizer = torch.optim.Adam(shader.parameters(), lr=self.shader_lr)
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)

        # Configure the renderer
        self.renderer = Renderer(device=device)
        self.renderer.set_near_far(self.views, torch.from_numpy(self.bbox.corners).to(device), epsilon=0.5)

        # Configure the view sampler
        self.view_sampler = ViewSampler(views=self.views, **ViewSampler.get_parameters(params))

        # Set up save paths
        run_name = params.run_name if params.run_name is not None else params.input_dir.parent.name
        experiment_dir = params.output_dir / run_name

        self.images_save_path = experiment_dir / "images"
        self.meshes_save_path = experiment_dir / "meshes"
        self.shaders_save_path = experiment_dir / "shaders"
        self._create_dirs()

    @staticmethod
    def load_views(input_dir, image_scale, pose_estimate=None):
        # Read the views
        if pose_estimate is None:
            views = read_views(input_dir, scale=image_scale,
                               device=device)
        else:
            raise NotImplementedError("need to add colmap pipeline")
        return views

    @staticmethod
    def load_mesh(initial_mesh_name, views, bbox):
        # Obtain the initial mesh and compute its connectivity
        if initial_mesh_name in mesh_generator_names:
            # Use args.initial_mesh as mesh generator name
            if bbox is None:
                raise RuntimeError("Generated meshes require a bounding box.")
            mesh = generate_mesh(initial_mesh_name, views, bbox, device=device)
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

    def visualize_views(self, save_path):
        visualize_views(self.views, show=False, save_path=save_path)

    def _normalize_views(self):
        # Apply the normalizing affine transform, which maps the bounding box to
        # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
        normalizer = self.space_normalization
        self.views = normalizer.normalize_views(self.views)
        self.mesh = normalizer.normalize_mesh(self.mesh)
        self.bbox = normalizer.normalize_aabb(self.bbox)


    def reconstruction_step(self):

        mesh = self.mesh.with_vertices(self.mesh.vertices + self.vertex_offsets)

        # Sample a view subset
        views_subset = self.view_sampler(self.views)

        # Render the mesh from the views
        # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
        gbuffers = self.renderer.render(
            views_subset, mesh,
            channels=['mask', 'position', 'normal'],
            with_antialiasing=True)

        loss = torch.tensor(0., device=device)

        # Combine losses and weights
        if (w := self.loss_weights['mask']) > 0:
            loss += mask_loss(views_subset, gbuffers) * w
        if (w := self.loss_weights['normal']) > 0:
            loss += normal_consistency_loss(mesh) * w
        if (w := self.loss_weights['laplacian']) > 0:
            loss += laplacian_loss(mesh) * w
        if (w := self.loss_weights['shading']) > 0:
            sh_loss = shading_loss(
                views_subset,
                gbuffers,
                shader=self.shader,
                shading_percentage=self.params.shading_percentage
            )
            loss += sh_loss * w

        # Optimize
        self.vertices_optimizer.zero_grad()
        self.shader_optimizer.zero_grad()
        loss.backward()
        self.vertices_optimizer.step()
        self.shader_optimizer.step()
        self.mesh = mesh
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

        self.mesh = Mesh(v_upsampled, f_upsampled, device=self.device)
        self.mesh.compute_connectivity()

        # Adjust weights and step size
        self.loss_weights['laplacian'] *= 4
        self.loss_weights['normal'] *= 4
        self.vertices_lr *= 0.75

        # Create a new optimizer for the vertex offsets
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)


    def visualize(self):
        with (torch.no_grad()):
            use_fixed_views = len(self.params.visualization_views) > 0
            if use_fixed_views:
                view_indices = self.params.visualization_views
            else:
                view_indices=[np.random.choice(list(range(len(self.views))))]
            for vi in view_indices:
                debug_view = self.views[vi]
                debug_gbuffer = self.renderer.render(
                    [debug_view],
                    self.mesh,
                    channels=['mask', 'position', 'normal'],
                    with_antialiasing=True)[0]
                position = debug_gbuffer["position"]
                normal = debug_gbuffer["normal"]
                view_direction = torch.nn.functional.normalize(
                    debug_view.camera.center - position, dim=-1)

                # Save the shaded rendering
                shaded_image = self.shader(position, normal, view_direction) * \
                               debug_gbuffer["mask"] + (1 - debug_gbuffer["mask"])
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
                                 device=device, dtype=torch.float32)
                normal_image = (0.5 * (
                            normal @ debug_view.camera.R.T @ R.T + 1)) * \
                               debug_gbuffer["mask"] + (1 - debug_gbuffer["mask"])
                plt.imsave(normal_path / f'neuralshading_{self.iteration}.png',
                           normal_image.cpu().numpy())

    def save_mesh(self):
        with torch.no_grad():
            mesh_for_writing = self.space_normalization.denormalize_mesh(
                self.mesh.detach().to('cpu'))
            write_mesh(self.meshes_save_path / f"mesh_{self.iteration:06d}.obj",
                       mesh_for_writing)
        self.shader.save(self.shaders_save_path / f'shader_{self.iteration:06d}.pt')

    def run_to_completion(self, total):
        progress_bar = tqdm(range(self.iteration, total + 1))
        for it in progress_bar:
            if it in self.params.upsample_iterations:
                self.upsample_mesh()
            step_loss = self.reconstruction_step()
            can_vis = self.params.visualization_frequency > 0 and shader is not None
            should_vis = it == 1 or it % self.params.visualization_frequency == 0
            if can_vis and should_vis:
                self.visualize()
            can_save = self.params.save_frequency > 0
            should_save = it == 1 or it % self.params.save_frequency == 0
            if can_save and should_save:
                self.save_mesh()
            self.iteration += 1
            progress_bar.set_postfix({'loss': step_loss})


    def recover(self):
        max_step = 0
        for fpath in self.meshes_save_path.iterdir():
            _, val = fpath.stem.split("_", maxsplit=1)
            max_step = max(max_step, int(val))
        mesh_path = self.meshes_save_path / f"mesh_{max_step:06d}.obj"
        shader_path = self.shaders_save_path / f"shader_{max_step:06d}.pt"
        self.iteration = max_step
        self.mesh = read_mesh(mesh_path, self.device)
        self.mesh = self.space_normalization.normalize_mesh(self.mesh)
        self.shader = NeuralShader.load(shader_path, self.device)

        self.shader_optimizer = torch.optim.Adam(shader.parameters(), lr=self.shader_lr)
        self.vertex_offsets = torch.zeros_like(self.mesh.vertices)
        self.vertex_offsets.requires_grad = True
        self.vertices_optimizer = torch.optim.Adam([self.vertex_offsets], lr=self.vertices_lr)
        print(f"recovering at iteration {self.iteration}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Multi-View Mesh Reconstruction with Neural Deferred Shading', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=Path, default="./data", help="Path to the input data")
    parser.add_argument('--input_bbox', type=Path, default=None, help="Path to the input bounding box. If None, it is computed from the input mesh")
    parser.add_argument('--pose_estimate', type=str, choices=["colmap"], default=None, help="If camera poses are unknown")
    parser.add_argument('--output_dir', type=Path, default="./out", help="Path to the output directory")
    parser.add_argument('--initial_mesh', type=str, default="vh32", help="Initial mesh, either a path or one of [vh16, vh32, vh64, sphere16]")
    parser.add_argument('--image_scale', type=int, default=1, help="Scale applied to the input images. The factor is 1/image_scale, so image_scale=2 halves the image size")
    parser.add_argument("--timeout", type=int, default=800, help="timeout for 1000 iterations to restart")
    parser.add_argument('--iterations', type=int, default=2000, help="Total number of iterations")
    parser.add_argument('--run_name', type=str, default=None, help="Name of this run")
    parser.add_argument('--lr_vertices', type=float, default=1e-3, help="Step size/learning rate for the vertex positions")
    parser.add_argument('--lr_shader', type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    parser.add_argument('--upsample_iterations', type=int, nargs='+', default=[500, 1000, 1500], help="Iterations at which to perform mesh upsampling")
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


    # Create the optimizer for the neural shader
    shader = NeuralShader(hidden_features_layers=args.hidden_features_layers,
                          hidden_features_size=args.hidden_features_size,
                          fourier_features=args.fourier_features,
                          activation=args.activation,
                          fft_scale=args.fft_scale,
                          last_activation=torch.nn.Sigmoid,
                          device=device)

    reconstructor = Reconstructor(shader, args, device=device)
    while True:
        try:
            training_job = Thread(target=reconstructor.run_to_completion, args=(args.iterations, ))
            training_job.daemon = True
            training_job.start()
            print("running job!")
            training_job.join(timeout=(args.timeout // args.image_scale))
            if training_job.is_alive():
                reconstructor.recover()
            break
        except KeyboardInterrupt:
            sys.exit()





