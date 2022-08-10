"""Definition of structured hydra config."""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from gaia.enums import Cadence


@dataclass
class DataFetchConfig:
    num_parallel_requests: int
    processed_urls_filename: str
    nasa_urls: list[str]
    mast_base_url: str


@dataclass
class DataConfig:
    cadence: Cadence
    root_dir: str
    time_series_dir: str
    tables_dir: str
    tce_filename: str
    fetch: DataFetchConfig


@dataclass
class BaseConfig:
    data: DataConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)
