import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Literal, Set, Any, Dict

from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.errors import ConfigAttributeError

from pydantic.dataclasses import dataclass
from pydantic import validator

from pydantic import BaseModel as PydanticBaseModel

import constants
from data import Dataset
from helpers import flatten
from processing.differential_analysis import differential_comparison
from visualization.abundance_bars import run_plot_abundance_bars

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
    """
    Sets default values or fills them from the method yaml file
    """

    barcolor: str = "timepoint"
    axisx: str = "condition"
    axisx_labeltilt: int = 20  # 0 is no tilt
    width_each_subfig: int = 3
    wspace_subfigs: int = 1

    def build(self) -> "AbundancePlot":
        return AbundancePlot(config=self)


class DifferentialAnalysisConfig(MethodConfig):
    """
    Sets default values or fills them from the method yaml file
    """

    grouping: ListConfig = ["condition", "timepoint"]
    qualityDistanceOverSpan: float
    correction_method: str = "bonferroni"
    impute_values: DictConfig

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
        logger.info("Will plot the abundance plot, with the following config: %s", self.config)
        dataset.load_compartmentalized_data(suffix=cfg.suffix)
        self.check_expectations(cfg, dataset)
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.makedirs(out_plot_dir, exist_ok=True)
        run_plot_abundance_bars(dataset, out_plot_dir, cfg)

    def check_expectations(self, cfg, dataset):
        # check that thresholds are provided in the config
        try:
            if not set(cfg.analysis.metabolites_to_plot.keys()).issubset(dataset.metadata_df["short_comp"]):
                raise ValueError(
                    f"[Analysis > Metabolites to plot > compartments] are missing from [Metadata > Compartments]"
                )
            if not set(cfg.analysis.time_sel).issubset(set(dataset.metadata_df["timepoint"])):
                raise ValueError(
                    f"[Analysis > Time sel] time points provided in the config file are not present in [Metadata > timepoint]"
                )
        except ConfigAttributeError as e:
            logger.error(f"Mandatory parameter not provided in the config file:{e}, aborting")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency:{e}")
            sys.exit(1)


class DifferentialAnalysis(Method):
    config: DifferentialAnalysisConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info(f"The current working directory is {os.getcwd()}")
        logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
        logger.info("Will perform differential analysis, with the following config: %s", self.config)
        dataset.load_compartmentalized_data(suffix=cfg.suffix)
        out_table_dir = os.path.join(os.getcwd(), cfg.table_path)
        os.makedirs(out_table_dir, exist_ok=True)
        self.check_expectations(cfg, dataset)
        for file_name, test in cfg.analysis.statistical_test.items():
            if test is None:
                continue
            logger.info(f"Running differential analysis of {dataset.get_file_for_label(file_name)} using {test} test")
            differential_comparison(file_name, dataset, cfg, test, out_table_dir=out_table_dir)

    def check_expectations(self, cfg, dataset):
        # check that thresholds are provided in the config
        try:
            float(cfg.analysis.thresholds.padj) is not None and float(
                cfg.analysis.thresholds.absolute_log2FC
            ) is not None
            if not set(cfg.analysis.metabolites_to_plot.keys()).issubset(dataset.metadata_df["short_comp"]):
                raise ValueError(
                    f"Comparisons > Conditions or timepoints provided in the config file are not present in the metadata file, aborting"
                )

            if not set(cfg.analysis.time_sel).issubset(set(dataset.metadata_df["timepoint"])):
                raise ValueError(
                    f"Timepoints provided in the config file are not present in the metadata file, aborting"
                )
            # all values in the comparisons arrays of arrays are in the metadata, either as conditions or timepoints
            conditions_and_tp = set(dataset.metadata_df["condition"]).union(set(dataset.metadata_df["timepoint"]))
            comparisons = set(flatten(cfg.analysis.comparisons))
            diff = comparisons.difference(conditions_and_tp)
            if not comparisons.issubset(conditions_and_tp):
                raise ValueError(
                    f"Comparisons > Conditions or timepoints provided in the config file {diff} are not present in the metadata file, aborting"
                )
            # comparison_mode is one of the constant values
            constants.assert_literal(cfg.analysis.comparison_mode, constants.comparison_modes_types, "comparison_mode")
        except ConfigAttributeError as e:
            logger.error(f"Mandatory parameter not provided in the config file:{e}, aborting")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency:{e}")
            sys.exit(1)
