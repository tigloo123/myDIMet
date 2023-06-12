import logging
import os
from pathlib import Path
from typing import Optional, List, Literal, Set, Any, Dict

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig

from pydantic.dataclasses import dataclass
from pydantic import validator

from pydantic import BaseModel as PydanticBaseModel

from data import Dataset
from visualization.abundance_bars import run_steps_abund_bars
from processing.differential_analysis import differential_comparison

logger = logging.getLogger(__name__)

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


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


class DifferentialAnalysisConfig(MethodConfig):
    '''
    Sets default values or fills them from the method yaml file
    '''
    comparison_mode:str
    comparisons: ListConfig = [["Control", "Condition"]]  # for each pair, last must be control
    correction_method:str
    grouping: ListConfig = ["condition", "timepoint"]
    impute_values : Dict[str, Optional[str]]
    qualityDistanceOverSpan:float
    statistical_test : Dict[str, Optional[str]]
    thresholds : Dict[str, Optional[float]]
    def build(self) -> "DifferentialAnalysis":
        return DifferentialAnalysis(config=self)

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
        dataset.load_compartmentalized_data(suffix=cfg.suffix)
        abundances_file_name = cfg.analysis.dataset.abundances_file_name
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.makedirs(out_plot_dir,exist_ok=True)
        run_steps_abund_bars(
            abundances_file_name,
            dataset,
            out_plot_dir,
            cfg)

class DifferentialAnalysis(Method):
    config: DifferentialAnalysisConfig
    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info(f"The current working directory is {os.getcwd()}")
        logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
        logger.info("Will perform differential analysis, with the following config: %s", self.config)
        dataset.load_compartmentalized_data(suffix=cfg.suffix)
        out_table_dir = os.path.join(os.getcwd(), cfg.table_path)
        os.makedirs(out_table_dir, exist_ok=True)

        for file_name, test in self.config.statistical_test.items():
            if test is None : continue
            logger.info(f"Running differential analysis on {file_name} using {test} test")
            differential_comparison(file_name, dataset, cfg, test,out_table_dir=out_table_dir)
