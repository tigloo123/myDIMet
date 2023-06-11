import logging
import os
from pathlib import Path
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


class Dataset(BaseModel):
    config: DatasetConfig
    # computed attributes
    metadata_df: pd.DataFrame = None
    sub_folder_absolute: str = None

    def preload(self):
        # check if we have a relative or absolute path, compute the absolute path then load the data using pandas
        # store the data in self.metadata
        if not self.config.subfolder.startswith("/"):
            self.sub_folder_absolute = os.path.join(Path(__file__).parent.parent.parent, "data", self.config.subfolder)
        else:
            self.sub_folder_absolute = self.self.config.subfolder
        self.metadata_df = pd.read_csv(os.path.join(self.sub_folder_absolute,self.config.metadata), sep="\t", header=0, index_col=0)
        logger.info("Loaded metadata: \n%s", self.metadata_df.head())
