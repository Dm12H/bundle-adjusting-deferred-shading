from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import re
import sys

from nds.utils.geometry import AABB
from nds.utils.idr import decompose



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Directory containing the DTU data in the IDR format.")
    parser.add_argument("output_dir", type=Path, help="Output directory for the DTU data in NDS format.")
    parser.add_argument("dtu_dir", type=Path, help="Path to dtu dataset ")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    dtu_dir: Path = args.dtu_dir

    id_regexp = re.compile(r"\d+")
    dtu_scans = [scan_dir.name for scan_dir in dtu_dir.iterdir() if scan_dir.is_dir()]
    id_to_dtu = {int(id_regexp.search(scan).group()): scan for scan in dtu_scans}

    for scan_dir in [directory for directory in input_dir.iterdir() if directory.is_dir()]:
        print(f"-- Converting {scan_dir.name}")
        idr_id = int(id_regexp.search(scan_dir.name).group())
        dtu_name = id_to_dtu.get(idr_id, None)
        if dtu_name is None:
            print(f"no corresponding dtu-nds scan for {scan_dir.name}")
            continue
        scan_output_dir = output_dir / dtu_name

        scan_output_dir.mkdir(parents=True, exist_ok=True)

        # Read cameras
        cameras = np.load(scan_dir / 'cameras_linear_init.npz')

        # Make a unit aabb 
        #! WARNING! This assumes a symmetric bbox (aka cube) if the bbox is not a cube 
        #! this will not give the bbox back, but something similar
        bbox = np.array([[-.5, -.5, -.5],
                         [.5,  .5,  .5]], dtype=np.float32)

        views_output_dir = scan_output_dir / "views"
        views_output_dir.mkdir(parents=True, exist_ok=True)
        im_paths = dtu_dir / dtu_name / "views"
        bbox_denormalized = None
        im_files = filter(lambda x: x.suffix == ".png", im_paths.iterdir())
        for i, view in enumerate(im_files):
            K, R, t = decompose(cameras[f"world_mat_{i}"][:3,:])
            np.savetxt(views_output_dir / f'dtu_cam{i:06d}_r.txt', R, fmt='%.20f')
            np.savetxt(views_output_dir / f'dtu_cam{i:06d}_t.txt', t, fmt='%.20f')
            np.savetxt(views_output_dir / f'dtu_cam{i:06d}_k.txt', K, fmt='%.20f')

            A_inv = cameras[f"scale_mat_{i}"]
            bbox_denormalized = (bbox @ A_inv[:3, :3].T) + A_inv[:3, 3][np.newaxis, :]

        # Save the bounding box
        # NOTE (MW): This assumes that all scale_mats are equal for a scan (which seems to be the case)
        #            Otherwise we would choose the last one, which seems a bit arbitrary
        if bbox_denormalized is None:
            raise ValueError("not a single relevant image in dtu dir was found")
        aabb = AABB(bbox_denormalized)
        aabb.save(scan_output_dir / "bbox_dtu.txt")