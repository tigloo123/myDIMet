import logging
import os
from pathlib import Path
from typing import Optional, List, Literal, Set

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from pydantic.dataclasses import dataclass
from pydantic import validator

from pydantic import BaseModel as PydanticBaseModel

from data import Dataset
from visualization.abundance_bars import run_steps_abund_bars


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


logger = logging.getLogger(__name__)


class MethodConfig(BaseModel):
    label: str
    name: str

    def build(self) -> "Method":
        raise NotImplementedError


class AbundancePlotConfig(MethodConfig):
    '''
    Sets default values or fills them from the method yaml file
    '''
    barcolor: str = "timepoint"
    axisx: str = "condition"
    axisx_labeltilt: int = 20  # 0 is no tilt
    width_each_subfig: int = 3
    wspace_subfigs: int = 1

    def build(self) -> "AbundancePlot":
        return AbundancePlot(config=self)


class Method(BaseModel):
    config: MethodConfig

    def plot(self):
        logger.info("Will plot the method, with the following config: %s", self.config)

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Not instantialted in the parent class.")
        raise NotImplementedError

class AbundancePlot(Method):
    config: AbundancePlotConfig
    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info(f"The current working directory is {os.getcwd()}")
        logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
        logger.info("Will plot the abundance plot, with the following config: %s", self.config)
        dataset.load_abundance_compartment_data(suffix=cfg.suffix)
        abund_tab_prefix = cfg.analysis.dataset.abundance_file_name
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.mkdir(out_plot_dir)
        run_steps_abund_bars(
            abund_tab_prefix,
            dataset,
            out_plot_dir,
            cfg)

