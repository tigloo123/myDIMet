import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from data import Dataset
from method import Method
from visualization.abundance_bars import run_steps_abund_bars

from data import make_dataset

logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main_run_analysis(cfg: DictConfig) -> None:
    logger.info(f"The current working directory is {os.getcwd()}")
    logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))

    dataset: Dataset = Dataset(config=hydra.utils.instantiate(cfg.analysis.dataset))
    dataset.preload()
    make_dataset.split_datasets(cfg, dataset)
    method: Method = hydra.utils.instantiate(cfg.analysis.method).build() # method factory

    method.run(cfg, dataset)


if __name__ == "__main__":
    main_run_analysis()