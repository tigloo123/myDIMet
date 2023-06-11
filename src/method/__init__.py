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


class MethodConfig(BaseModel):
    label: str
    name: str

    def build(self) -> "Method":
        raise NotImplementedError


class AbundancePlotConfig(MethodConfig):
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


class AbundancePlot(Method):
    config: AbundancePlotConfig

    def plot(self):
        logger.info("Will plot the abundance plot, with the following config: %s", self.config)
