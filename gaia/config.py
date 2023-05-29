from pathlib import Path
from typing import Any

from pydantic import BaseModel, DirectoryPath, FilePath, HttpUrl, PositiveInt

from gaia.enums import Cadence


class DataApiConfig(BaseModel):
    mast_base_url: HttpUrl
    nasa_base_url: HttpUrl
    num_async_requests: PositiveInt
    requests: list[tuple[HttpUrl, Path]]


class DataLoadConfig(BaseModel):
    skip_tables: bool
    observation_cadence: Cadence
    tce_file: Path
    target_id_column: str
    api: DataApiConfig
    verbose: bool


class TceMergeConfig(BaseModel):
    select_sql: str
    join_sql: str
    label_conditions_case_sql: list[str]
    output_file: Path


class FileConvertConfig(BaseModel):
    inputs: FilePath | str
    output: Path
    columns: list[str]
    columns_map: dict[str, str]


class TimeSeriesConvertConfig(BaseModel):
    inputs: FilePath | str
    output: Path
    data_header: str
    data_columns: list[str]
    meta_columns: list[str]
    names_map: dict[str, str]
    path_target_id_pattern: str


class DataConvertConfig(BaseModel):
    tce: FileConvertConfig
    stellar: FileConvertConfig
    time_series: TimeSeriesConvertConfig


class DataConfig(BaseModel):
    raw_tables_dir: Path
    raw_time_series_dir: Path
    interim_tables_dir: Path
    interim_time_series_dir: Path
    load: DataLoadConfig
    convert: DataConvertConfig
    tce_merge: TceMergeConfig


class UiConfig(BaseModel):
    assets_dir: DirectoryPath
    external_stylesheets: list[HttpUrl]
    available_graphs: dict[str, str]
    tce_filepath: FilePath
    stellar_parameters_filepath: FilePath
    time_series_dir: DirectoryPath
    server_params: dict[str, Any]


class Config(BaseModel):
    mission: str
    data: DataConfig
    ui: UiConfig
