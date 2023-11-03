from ipaddress import IPv4Address
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    HttpUrl,
    PositiveInt,
    StringConstraints,
)
from pydantic.dataclasses import dataclass

from gaia.data.converters import FitsConvertingSettings
from gaia.data.models import NasaTableRequest
from gaia.enums import Cadence
from gaia.io import Columns


NonEmptyString = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class ImmutableBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)


def create_dir_if_not_exist(path_str: str) -> Path:
    path = Path(path_str)

    if not path.exists():
        path.mkdir(parents=True)

    return path


ExistentDirectoryPath = Annotated[DirectoryPath, BeforeValidator(create_dir_if_not_exist)]


class DataConfig(ImmutableBaseModel):
    raw_time_series_dir: ExistentDirectoryPath
    raw_tables_dir: ExistentDirectoryPath
    raw_targets_filepath: Path
    raw_tce_filepath: Path
    raw_stellar_params_filepath: Path
    raw_false_positive_filepath: Path
    interim_time_series_path: NonEmptyString
    interim_tce_filepath: NonEmptyString
    interim_stellar_params_filepath: NonEmptyString
    final_path: NonEmptyString
    dashboard_assets_dir: ExistentDirectoryPath


# NOTE: @dataclass when '_target_' or '_partial_' is present because Pydantic excludes these fields
# See: https://docs.pydantic.dev/latest/usage/models/#automatically-excluded-attributes


@dataclass(frozen=True)
class TargetableObject:
    _target_: NonEmptyString


@dataclass(frozen=True)
class SaverConfig(TargetableObject):
    tables_dir: DirectoryPath
    time_series_dir: DirectoryPath


@dataclass(frozen=True)
class DownloaderConfig(TargetableObject):
    saver: SaverConfig
    cadence: Cadence
    num_async_requests: PositiveInt


class DownloadConfig(ImmutableBaseModel):
    script_description: NonEmptyString
    tce_filepath: Path
    tce_target_id_column: NonEmptyString
    download_tables: bool
    download_time_series: bool
    downloader: DownloaderConfig
    tables: list[NasaTableRequest]


# Use discriminant unions to chose correct converter config
# See: https://docs.pydantic.dev/latest/api/standard_library_types/#discriminated-unions-aka-tagged-unions
@dataclass(frozen=True)
class CsvConverterConfig(TargetableObject):
    converter_type: Literal["csv"]
    include_columns: Columns | None = None


@dataclass(frozen=True)
class FitsConverterConfig(TargetableObject):
    converter_type: Literal["fits"]
    settings: FitsConvertingSettings


class ConversionArgs(ImmutableBaseModel):
    inputs: FilePath | str
    output: Path | str


class ConversionConfig(ImmutableBaseModel):
    converter: CsvConverterConfig | FitsConverterConfig = Field(discriminator="converter_type")
    args: ConversionArgs


class TceMergeConfig(ImmutableBaseModel):
    select_sql: NonEmptyString
    join_sql: NonEmptyString
    labels_case_sql: list[NonEmptyString]
    output: Path
    label_column: NonEmptyString


class PreprocessingConfig(ImmutableBaseModel):
    tce_merge: TceMergeConfig
    script_description: NonEmptyString
    tce_conversion: ConversionConfig
    stellar_params_conversion: ConversionConfig
    time_series_conversion: ConversionConfig


Port = Annotated[int, Field(gt=0, le=65535)]


class DashServerParameters(ImmutableBaseModel):
    debug: bool
    host: Annotated[str, AfterValidator(lambda host: str(IPv4Address(host)))]
    port: Port
    dev_tools_hot_reload: bool


class DataStoreConfig(ImmutableBaseModel):
    time_series: dict[str, Any]
    tce: dict[str, Any]
    stellar_params: dict[str, Any]


class UIConfig(ImmutableBaseModel):
    script_description: NonEmptyString
    assets_dir: DirectoryPath
    external_stylesheets: list[HttpUrl]
    available_graphs: dict[str, str]
    server_params: DashServerParameters

    # Should be hydra partial or targetable objects
    graphs_preprocessors: dict[str, Any]

    stellar_parameters_units: dict[str, str]
    planetary_parameters_units: dict[str, str]


# NOTE: Keep in-sync with: https://cloud.google.com/compute/docs/regions-zones#available
GCPRegion = Literal[
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast2",
    "asia-northeast3",
    "asia-south1",
    "asia-south2",
    "asia-southeast1",
    "asia-southeast2",
    "australia-southeast1",
    "australia-southeast2",
    "europe-central2",
    "europe-north1",
    "europe-southwest1",
    "europe-west1",
    "europe-west10",
    "europe-west12",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west8",
    "europe-west9",
    "me-central1",
    "me-central2",
    "me-west1",
    "northamerica-northeast1",
    "northamerica-northeast2",
    "southamerica-east1",
    "southamerica-west1",
    "us-central1",
    "us-east1",
    "us-east4",
    "us-east5",
    "us-south1",
    "us-west1",
    "us-west2",
    "us-west3",
    "us-west4",
]


class RuntimeConfig(BaseModel):
    container_image: str
    version: str
    properties: dict[str, str] = Field(default_factory=dict)


class DataprocServerlessConfig(BaseModel):
    runtime_config: RuntimeConfig
    subnetwork_uri: str
    main_script_uri: str
    region: GCPRegion


class SparkConfig(ImmutableBaseModel):
    app_name: NonEmptyString
    cache_input_data: bool
    config_properties: dict[str, str] = Field(default_factory=dict)
    """Spark configuration properties.

    See: https://spark.apache.org/docs/latest/configuration.html#available-properties
    """


class CreateFeaturesConfig(ImmutableBaseModel):
    script_description: NonEmptyString
    spark: SparkConfig
    script_path: FilePath
    seed: PositiveInt
    shuffle_dataset: bool
    test_size: Annotated[float, Field(gt=0, le=1)]
    validation_size: Annotated[float, Field(gt=0, le=1)]
    tce_target_id_column: NonEmptyString
    stellar_params_target_id_column: NonEmptyString
    tce_excluded_columns: list[NonEmptyString]
    stellar_params_excluded_columns: list[NonEmptyString]
    sigma_cut: tuple[float, float] | float
    tce_key_by: list[NonEmptyString]
    stellar_params_key_by: list[NonEmptyString]
    time_series_id_key: NonEmptyString
    time_key: NonEmptyString
    flux_key: NonEmptyString
    centroid_x_key: NonEmptyString
    centroid_y_key: NonEmptyString
    output: Path | str
    flatten_func: dict[str, Any]
    generate_local_view_func: dict[str, Any]
    generate_global_view_func: dict[str, Any]
    tce_source: str
    stellar_params_source: str
    time_series_source: str
    tce_mapper: dict[str, Any]
    stellar_params_mapper: dict[str, Any]
    time_series_mapper: dict[str, Any]
    input_format: Literal["json", "parquet"]  # TODO: do osobnego pliku
    output_format: Literal["json", "parquet", "tfrecord"]
    partition_by: list[NonEmptyString]
    dataproc: DataprocServerlessConfig | None = None
    """Google Cloud Dataproc Serverless batch workload config. If not set assume local mode."""


class AppConfig(ImmutableBaseModel):
    data: DataConfig
    download: DownloadConfig
    preprocess_data: PreprocessingConfig
    ui: UIConfig
    data_providers: DataStoreConfig
    create_features: CreateFeaturesConfig
