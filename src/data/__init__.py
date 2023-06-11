import logging
import os
from pathlib import Path
from typing import Optional, List, Literal, Set

import pandas as pd

from pydantic.dataclasses import dataclass
from pydantic import validator

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

    name_abundance: str = "AbundanceCorrected"
    name_meanE_or_fracContrib: str = "MeanEnrichment13C"
    name_isotopologue_prop: str = "IsotopologuesProp"  # isotopologue proportions
    name_isotopologue_abs: str = "IsotopologuesAbs"  # isotopologue absolute values


class Dataset(BaseModel):
    config: DatasetConfig
    raw_data_folder: str = None
    processed_data_folder: str = None
    sub_folder_absolute: str = None
    df_metadata: Optional[pd.DataFrame] = None
    df_abundance: Optional[pd.DataFrame] = None
    df_meanE_or_fracContrib: Optional[pd.DataFrame] = None
    df_isotopologue_prop: Optional[pd.DataFrame] = None
    df_isotopologue_abs: Optional[pd.DataFrame] = None
    available_datasets: Set[Literal["metadata", "abundance", "meanE_or_fracContrib", "isotopologue_prop", "isotopologue_abs"]] = set()

    def preload(self):
        # check if we have a relative or absolute path, compute the absolute path then load the data using pandas
        # store the data in self.metadata
        if not self.config.subfolder.startswith("/"):
            self.sub_folder_absolute = os.path.join(Path(__file__).parent.parent.parent, "data", self.config.subfolder)
            self.raw_data_folder = os.path.join(self.sub_folder_absolute, 'raw')
            self.processed_data_folder = os.path.join(self.sub_folder_absolute, 'processed')
        else:
            self.sub_folder_absolute = self.self.config.subfolder
        # start loading the dataframes
        file_paths = [
            ("metadata", os.path.join(self.raw_data_folder, self.config.metadata + ".csv")),
            ("abundance", os.path.join(self.raw_data_folder, self.config.name_abundance + ".csv")),
            ("meanE_or_fracContrib", os.path.join(self.raw_data_folder, self.config.name_meanE_or_fracContrib + ".csv")),
            ("isotopologue_prop", os.path.join(self.raw_data_folder, self.config.name_isotopologue_prop + ".csv")),
            ("isotopologue_abs", os.path.join(self.raw_data_folder, self.config.name_isotopologue_abs + ".csv"))
        ]
        dfs = []
        for label, file_path in file_paths:
            try:
                dfs.append(pd.read_csv(file_path, sep="\t", header=0, index_col=0))
                self.available_datasets.add(label)
            except FileNotFoundError:
                dfs.append(None)
            except Exception as e:
                logger.error("Failed to load file %s during preload, aborting", file_path)
                raise e

        self.df_metadata, self.df_abundance, self.df_meanE_or_fracContrib, self.df_isotopologue_prop, self.df_isotopologue_abs = dfs

        # log the first 5 rows of the metadata
        logger.info("Loaded metadata: \n%s", self.df_metadata.head())
        logger.info("Finished loading dataset %s, available dataframes are : %s", self.config.label, self.available_datasets)
