import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from data import Dataset
from visualization.abundance_bars import run_steps_abund_bars

from data import make_dataset

logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main_abubdance_plot(cfg: DictConfig):
    logger.info(f"The current working directory is {os.getcwd()}")
    logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
    dataset: Dataset = Dataset(config=hydra.utils.instantiate(cfg.analysis.dataset))
    dataset.preload()
    dataset.load_abundance_compartment_data(suffix=cfg.suffix)
    abund_tab_prefix = cfg.analysis.dataset.abundance_file_name
    out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
    os.mkdir(out_plot_dir)
    run_steps_abund_bars(
        abund_tab_prefix,
        dataset,
        out_plot_dir,
        cfg)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main_split_datasets(cfg: DictConfig):
    logger.info(f"The current working directory is {os.getcwd()}")
    logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
    dataset: Dataset = Dataset(config=hydra.utils.instantiate(cfg.analysis.dataset))
    dataset.preload()

    out_data_path = os.path.join(os.getcwd(), cfg.output_path)
    os.mkdir(out_data_path)
    make_dataset.save_datafiles_split_by_compartment(cfg, dataset = dataset, out_data_path = out_data_path)


if __name__ == "__main__":

    main_split_datasets()
    #main_abubdance_plot()
