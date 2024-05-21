from reconstruct import Reconstructor
from omegaconf import OmegaConf
from utils.type_checking import PathConfig
import torch


def run_exp() -> None:
    paths_schema = OmegaConf.structured(PathConfig)
    cfg = OmegaConf.load("params.yaml")
    print(cfg)
    pathconfig = OmegaConf.merge(paths_schema, cfg.paths)
    cfg.paths = pathconfig
    device = torch.device('cpu')
    if torch.cuda.is_available() and cfg.run.device >= 0:
        device = torch.device(f'cuda:{cfg.run.device}')
    rec = Reconstructor(cfg, device)
    rec.run_to_completion()


if __name__ == "__main__":
    run_exp()