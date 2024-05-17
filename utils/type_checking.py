from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PathConfig:
    input_dir: Optional[Path]
    input_bbox: Optional[Path]
    output_dir: Optional[Path]
    gt_points: Optional[Path]
    gt_masks: Optional[Path]

    def __init__(self,
                 input_dir=None,
                 input_bbox=None,
                 output_dir=None,
                 gt_points=None,
                 gt_masks=None):
        self.input_dir = Path(input_dir) if input_dir is not None else None
        self.input_bbox = Path(input_bbox) if input_bbox is not None else None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.gt_points = Path(gt_points) if gt_points is not None else None
        self.gt_masks = Path(gt_masks) if gt_masks is not None else None
