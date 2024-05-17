from pathlib import Path
from PIL import Image
import pycolmap


image_dir = Path("/home/dm12h/data/data/65_skull/views")
output_path = Path("out", image_dir.parent.stem, "colmap")
exts = Image.registered_extensions()
supported_extensions = {ex for ex, f in exts.items() if f in Image.OPEN}
if not (mask_path := (image_dir.parent / "masks")).exists():
    mask_path.mkdir()
    for im_path in image_dir.iterdir():
        if im_path.suffix not in supported_extensions:
            continue
        im = Image.open(im_path)
        *_, alpha_channel = im.split()
        alpha_channel.save(mask_path / (im_path.name + ".png"))

output_path.mkdir(exist_ok=True)
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

imagereader_opts = pycolmap.ImageReaderOptions(mask_path=mask_path)
pycolmap.extract_features(database_path, image_dir, reader_options=imagereader_opts)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
