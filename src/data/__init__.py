import logging
import os
from pathlib import Path
from typing import Optional, List, Literal, Set, Dict

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import ListConfig
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    label: str
    name: str
    subfolder: str
    metadata: str
    # We allow for some default values for the following files
    # will be ignored if they do not exist

    # First condition is the reference condition (control)
    # Conditions should be a subset of the medata corresponding column
    conditions: ListConfig
    abundances_file_name: str = "AbundanceCorrected"
    meanE_or_fracContrib_file_name: str = "MeanEnrichment13C"
    isotopologue_prop_file_name: str = "IsotopologuesProp"  # isotopologue proportions
    isotopologue_abs_file_name: str = "IsotopologuesAbs"  # isotopologue absolute values

    def build(self) -> "Dataset":
        return Dataset(config=self)


class Dataset(BaseModel):
    config: DatasetConfig
    raw_data_folder: str = None
    processed_data_folder: str = None
    sub_folder_absolute: str = None
    metadata_df: Optional[pd.DataFrame] = None
    abundance_df: Optional[pd.DataFrame] = None
    meanE_or_fracContrib_df: Optional[pd.DataFrame] = None
    isotopologue_prop_df: Optional[pd.DataFrame] = None
    isotopologue_abs_df: Optional[pd.DataFrame] = None
    available_datasets: Set[
        Literal["metadata", "abundance", "meanE_or_fracContrib", "isotopologue_prop", "isotopologue_abs"]
    ] = set()
    compartmentalized_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}

    def preload(self):
        # check if we have a relative or absolute path, compute the absolute path then load the data using pandas
        # if the path is relative, we assume it is relative to the original CWD (befre hydra changed it)
        # store the data in self.metadata
        original_cwd = HydraConfig.get().runtime.cwd
        logger.info("Current config directory is %s", original_cwd)
        if not self.config.subfolder.startswith("/"):
            self.sub_folder_absolute = os.path.join(Path(original_cwd), "data", self.config.subfolder)
            logger.info("looking for data in %s", self.sub_folder_absolute)
        else:
            self.sub_folder_absolute = self.self.config.subfolder
        self.raw_data_folder = os.path.join(self.sub_folder_absolute, "raw")
        self.processed_data_folder = os.path.join(self.sub_folder_absolute, "processed")
        # start loading the dataframes
        file_paths = [
            ("metadata", os.path.join(self.raw_data_folder, self.config.metadata + ".csv")),
            ("abundance", os.path.join(self.raw_data_folder, self.config.abundances_file_name + ".csv")),
            (
                "meanE_or_fracContrib",
                os.path.join(self.raw_data_folder, self.config.meanE_or_fracContrib_file_name + ".csv"),
            ),
            ("isotopologue_prop", os.path.join(self.raw_data_folder, self.config.isotopologue_prop_file_name + ".csv")),
            ("isotopologue_abs", os.path.join(self.raw_data_folder, self.config.isotopologue_abs_file_name + ".csv")),
        ]
        dfs = []
        for label, file_path in file_paths:
            try:
                dfs.append(pd.read_csv(file_path, sep="\t", header=0))
                self.available_datasets.add(label)
            except FileNotFoundError:
                logger.critical("File %s not found, continuing, but this might fail miserably", file_path)
                dfs.append(None)
            except Exception as e:
                logger.error("Failed to load file %s during preload, aborting", file_path)
                raise e

        (
            self.metadata_df,
            self.abundance_df,
            self.meanE_or_fracContrib_df,
            self.isotopologue_prop_df,
            self.isotopologue_abs_df,
        ) = dfs

        # log the first 5 rows of the metadata
        logger.info("Loaded metadata: \n%s", self.metadata_df.head())
        logger.info(
            "Finished loading raw dataset %s, available dataframes are : %s", self.config.label, self.available_datasets
        )
        self.check_expectations()

    def check_expectations(self):
        # conditions should be a subset of the metadata corresponding column
        if not set(self.config.conditions).issubset(set(self.metadata_df["condition"].unique())):
            logger.error("Conditions %s are not a subset of the metadata declared conditions", self.config.conditions)
            raise ValueError(
                f"Conditions {self.config.conditions} are not a subset of the metadata declared conditions"
            )

    def load_compartmentalized_data(self) -> pd.DataFrame:
        compartments = self.metadata_df["short_comp"].unique().tolist()
        for c in compartments:
            file_paths = [
                (
                    "abundances_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.abundances_file_name}--{c}.tsv"),
                ),
                (
                    "meanE_or_fracContrib_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.meanE_or_fracContrib_file_name}--{c}.tsv"),
                ),
                (
                    "isotopologue_prop_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.isotopologue_prop_file_name}--{c}.tsv"),
                ),
                (
                    "isotopologue_abs_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.isotopologue_abs_file_name}--{c}.tsv"),
                ),
            ]

            for label, fp in file_paths:
                if label not in self.compartmentalized_dfs:
                    self.compartmentalized_dfs[label] = {}
                self.compartmentalized_dfs[label][c] = pd.read_csv(fp, sep="\t", header=0, index_col=0)
                logger.info("Loaded compartmentalized %s DF for %s", label, c)

    def get_file_for_label(self, label):
        if label == "abundances_file_name":
            return self.config.abundances_file_name
        elif label == "meanE_or_fracContrib_file_name":
            return self.config.meanE_or_fracContrib_file_name
        elif label == "isotopologue_prop_file_name":
            return self.config.isotopologue_prop_file_name
        elif label == "isotopologue_abs_file_name":
            return self.config.isotopologue_abs_file_name
        else:
            raise ValueError(f"Unknown label {label}")

    def load_compartmentalized_data_version2(self) -> pd.DataFrame:
        # version above had problem with loading more than one compartment for each label:
        # only one compartment when two available. This version2 corrects the problem

        compartments = self.metadata_df["short_comp"].unique().tolist()
        for label in ["abundances_file_name", "meanE_or_fracContrib_file_name",
                      "isotopologue_prop_file_name", "isotopologue_abs_file_name"]:
            self.compartmentalized_dfs[label] = {}
        for c in compartments:
            file_paths = [
                (
                    "abundances_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.abundances_file_name}--{c}.tsv"),
                ),
                (
                    "meanE_or_fracContrib_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.meanE_or_fracContrib_file_name}--{c}.tsv"),
                ),
                (
                    "isotopologue_prop_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.isotopologue_prop_file_name}--{c}.tsv"),
                ),
                (
                    "isotopologue_abs_file_name",
                    os.path.join(self.processed_data_folder, f"{self.config.isotopologue_abs_file_name}--{c}.tsv"),
                ),
            ]

            for label, fp in file_paths:
                if os.path.exists(fp):
                    self.compartmentalized_dfs[label][c] = pd.read_csv(fp, sep="\t", header=0, index_col=0)
                logger.info("Loaded compartmentalized %s DF for %s", label, c)
