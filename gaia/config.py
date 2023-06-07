from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, DirectoryPath, HttpUrl, PositiveInt, validator
from pydantic.dataclasses import dataclass
from pydantic.generics import GenericModel

from gaia.data.converters import FitsConvertingSettings
from gaia.downloaders import HTTPFileRequest
from gaia.enums import Cadence, Mission
from gaia.io import create_dir_if_not_exist


# NOTE: @dataclass when '_target_' or '_partial_' is present because Pydantic excludes these fields
# See: https://docs.pydantic.dev/latest/usage/models/#automatically-excluded-attributes


@dataclass
class SaverConfig:
    _target_: str
    tables_dir: str
    time_series_dir: str


@dataclass
class DataDownloaderConfig:
    _target_: str
    saver: SaverConfig
    cadence: Cadence
    nasa_base_url: HttpUrl
    mast_base_url: HttpUrl
    num_async_requests: PositiveInt

    class Config:
        arbitrary_types_allowed = True


class DataDownloadConfig(BaseModel):
    verbose: bool
    download_tables: bool
    download_time_series: bool
    downloader: DataDownloaderConfig
    tables_requests: list[HTTPFileRequest]
    script_description: str
    tce_file_path: Path
    tce_file_target_id_column: str

    class Config:
        arbitrary_types_allowed = True


class TceMergeConfig(BaseModel):
    script_description: str
    select_sql: str
    join_sql: str
    labels_case_sql: list[str]
    output: Path
    label_column: str


@dataclass
class TargetableObject:
    _target_: str


@dataclass
class BasicConverter(TargetableObject):
    ...


@dataclass
class TimeSeriesConverter(BasicConverter):
    settings: FitsConvertingSettings


TConverter = TypeVar("TConverter", bound=BasicConverter)


class ConverterConfig(GenericModel, Generic[TConverter]):
    converter: TConverter
    conversion_parameters: dict[str, Any]  # In the future, it may be some generic config


class ConversionConfig(BaseModel):
    tce: ConverterConfig[BasicConverter]
    stellar_parameters: ConverterConfig[BasicConverter]
    time_series: ConverterConfig[TimeSeriesConverter]


class PreprocessingConfig(BaseModel):
    tce_merge: TceMergeConfig
    conversion: ConversionConfig


class AppConfig(BaseModel):
    mission: Mission
    raw_tables_dir: DirectoryPath
    raw_time_series_dir: DirectoryPath
    interim_tables_dir: DirectoryPath
    interim_time_series_dir: DirectoryPath
    download: DataDownloadConfig
    preprocessing: PreprocessingConfig

    @validator(
        "raw_tables_dir",
        "raw_time_series_dir",
        "interim_tables_dir",
        "interim_time_series_dir",
        pre=True,
    )
    def create_dir_if_not_exist(cls, path_like: Any) -> Path:
        return create_dir_if_not_exist(path_like)
