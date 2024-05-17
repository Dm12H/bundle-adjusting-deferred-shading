from reconstruct import Reconstructor
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from utils.type_checking import PathConfig
import torch

cs = ConfigStore.instance()
cs.store(name="base_paths", group="paths", node=PathConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_exp(cfg : DictConfig) -> None:
    device = torch.device('cpu')
    if torch.cuda.is_available() and cfg.run.device >= 0:
        device = torch.device(f'cuda:{cfg.run.device}')
    rec = Reconstructor(cfg, device)
    rec.run_to_completion()

if __name__ == "__main__":
    run_exp()