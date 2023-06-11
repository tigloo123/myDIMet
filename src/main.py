import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from data import Dataset
from visualization.abundance_bars import run_steps_abund_bars

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"The current working directory is {os.getcwd()}")
    logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
    dataset: Dataset = Dataset(config=hydra.utils.instantiate(cfg.analysis.dataset))
    dataset.preload()
    dataset.load_abundance_compartment_data(suffix=cfg.suffix)
    abund_tab_prefix = cfg.analysis.dataset.name_abundance
    out_plot_dir = os.path.join(os.getcwd(), cfg.output_path)
    os.mkdir(out_plot_dir)
    run_steps_abund_bars(
        abund_tab_prefix,
        dataset,
        out_plot_dir,
        cfg)


if __name__ == "__main__":
    # parser = bars_args()
    # args = parser.parse_args()
    # process(args)
    main()
