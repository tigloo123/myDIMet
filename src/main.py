import hydra
import os

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import logging

from data import Dataset, DatasetConfig
from dbs import PostgreSQLConnection
from method import Method

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def func(cfg: DictConfig):
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    print(OmegaConf.to_yaml(cfg))
    # To access elements of the config
    print(f"The batch size is {cfg.batch_size}")
    print(f"The learning rate is {cfg['lr']}")
    logger.info("Hello from main.py")
    logger.warning("warning from main.py")
    connection = hydra.utils.instantiate(cfg.db)
    connection.connect()
    dataset: Dataset = Dataset(config=hydra.utils.instantiate(cfg.analysis.dataset))
    dataset.preload()
    method: Method = hydra.utils.instantiate(cfg.analysis.method).build()
    method.plot()


if __name__ == "__main__":
    func()
