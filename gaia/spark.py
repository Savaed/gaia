import re
from collections.abc import Callable, Hashable, Iterable, Sized
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

import hydra
import numpy as np
import pandas as pd
from google.cloud import storage
from pandas import Series
from pyspark import RDD
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from gaia.config import CreateFeaturesConfig, SparkConfig
from gaia.data.mappers import Mapper, map_dataclass_to_flat_dict
from gaia.data.models import TCE, Id, IterableOfSeries, ListOfSeries, PeriodicEvent
from gaia.data.preprocessing import (
    AggregateFunction,
    BinFunction,
    BinWidthFunction,
    SecondaryTransitAdjustedPadding,
    Spline,
    TimeBoundariesFunction,
    ViewGenerator,
    compute_euclidean_distance,
    flatten_time_series,
    phase_fold_time,
    remove_events,
)
from gaia.data.stores import TStellarParameters, TTce, TTimeSeries
from gaia.progress import ProgressBar
from gaia.utils import get_chunks


# from gaia.log import logger


T = TypeVar("T")

TimeWithSeries: TypeAlias = tuple[Series, Series]
FlattenFunction: TypeAlias = Callable[..., TimeWithSeries]
GenerateViewFunction: TypeAlias = Callable[..., list[tuple[Id, Series]]]


def normalize_data_frame_median_std(
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    excluded: Iterable[str] | None = None,
    sigma_cut: tuple[float, float] | float | None = None,
) -> pd.DataFrame:
    """Normalize using median and standard deviation.

    Normalization: `(X - median(X)) / std(X)`.

    Args:
        all_df (pd.DataFrame): All data
        train_df (pd.DataFrame): Training data on which compute `median` and `std`
        excluded (Iterable[str] | None, optional): Columns to exclude from normalization.
        Defaults to None.
        sigma_cut (tuple[float, float] | float | None, optional): Clip values beyond
        `sigma_cut * std(train_df)`. For `tuple` it is unpacked as `(lower_bound, upper_bound)`.
        For `float` it is `(-sigma_cut, sigma_cut)` Defaults to None.

    Returns:
        pd.DataFrame: Normalized `pd.DataFrame`
    """
    excluded = excluded or []

    # Preserve columns order
    included_columns = [col for col in all_df.columns if col not in excluded]

    train_df_copy = train_df[included_columns]
    median_df = train_df_copy.median()
    std_df = train_df_copy.std()
    df_pre_normalization = all_df[included_columns]

    normalized_df = (df_pre_normalization - median_df) / std_df

    if sigma_cut:
        std = normalized_df.std()

        if isinstance(sigma_cut, tuple):
            lower_bound, upper_bound = sigma_cut
            clip_limit = (lower_bound * std, upper_bound * std)
        else:
            clip_limit = (-sigma_cut * std, sigma_cut * std)

        normalized_df = normalized_df.clip(*clip_limit, axis=1)

    for col in excluded:
        normalized_df[col] = all_df[col]

    return normalized_df


def read_objects(
    spark: SparkSession,
    source: str | Iterable[str],
    mapper: Mapper[T],
    format: Literal["parquet", "json"],  # TODO: do osobnego pliku
) -> RDD[T]:
    """Read data from the source and map it to Python objects.

    Args:
        spark (SparkSession): Spark session
        source (str | Iterable[str]): URI(s) or path(s) to the data. Can point to the directory
        mapper (Mapper[T]): Mapper to map from read raw data to the output Python objects
        format (DataFormat): Format of source data

    Returns:
        RDD[T]: Spark RDD with read objects mapped.
    """
    return spark.read.format(format).load(source).rdd.map(mapper)


def generate_view(
    data: tuple[TimeWithSeries, Iterable[tuple[Id, PeriodicEvent]]],
    num_bins: int,
    time_boundaries_func: TimeBoundariesFunction,
    bin_width_func: BinWidthFunction,
    bin_func: BinFunction,
    aggregate_func: AggregateFunction,
    default: float | AggregateFunction,
) -> list[tuple[Id, Series]]:
    """Generate time series view for given TCEs.

    Args:
        data (tuple[TimeWithSeries, Iterable[tuple[Id, PeriodicEvent]]]): Time,
        values, TCE ID, transit event and secondary phase
        num_bins (int): Number of bins to include in the view
        time_boundaries_func (TimeBoundariesFunction): Function to compute min and max time for the
        view
        bin_width_func (BinWidthFunction): Function to compute bins width
        bin_func (BinFunction): Function to divide time series into `num_bins` bins
        aggregate_func (AggregateFunction): Function to aggregate values in each bins
        default (float | AggregateFunction): Default value or function for empty bins

    Returns:
        list[tuple[Id, Series]]: A list of views for a particiular TCE in form of (TCE ID, view)
    """
    (time, values), events = data
    views: list[tuple[Id, Series]] = []

    for tce_id, event in events:
        folded_time = phase_fold_time(time, epoch=event.epoch, period=event.period)

        # Sort time and values after each phase folding
        ind = np.argsort(folded_time)
        folded_time = folded_time[ind]
        sorted_values = values[ind]

        time_min_max = time_boundaries_func(event.period, event.duration)
        bin_width = bin_width_func(event.period, event.duration)
        generator = ViewGenerator(folded_time, sorted_values, bin_func, aggregate_func, default)

        view = generator.generate(num_bins, time_min_max, bin_width)
        views.append((tce_id, view))

    return views


def split_periods_to_odd_even(
    data: tuple[TimeWithSeries, Iterable[tuple[Id, PeriodicEvent]]],
) -> tuple[TimeWithSeries, TimeWithSeries]:
    """Split time series for odd and even periods.

    Args:
        data (tuple[TimeWithSeries, Iterable[tuple[Id, PeriodicEvent]]]): Time, values, iterable of
        TCE ID and transit event

    Returns:
        tuple[TimeWithSeries, TimeWithSeries]: Odd and even time series
    """
    odd_periods: list[TimeWithSeries] = []
    even_periods: list[TimeWithSeries] = []
    period_indices = []

    (time, values), events = data
    time_max = np.nanmax(time)

    for _, event in events:
        for period_end in np.arange(event.epoch + (event.period / 2), time_max, event.period):
            # The index of the point closest to the period end
            period_end_index = np.nanargmin(np.abs(time - period_end))
            period_indices.append(period_end_index)

        periods_time = np.array_split(time, period_indices)
        periods_values = np.array_split(values, period_indices)

        odd_periods_time = np.concatenate(periods_time[::2])
        even_periods_time = np.concatenate(periods_time[1::2])

        odd_periods_values = np.concatenate(periods_values[::2])
        even_periods_values = np.concatenate(periods_values[1::2])

        odd_periods.append((odd_periods_time, odd_periods_values))
        even_periods.append((even_periods_time, even_periods_values))

    odd_time = np.concatenate([time for time, _ in odd_periods])
    odd_values = np.concatenate([values for _, values in odd_periods])

    even_time = np.concatenate([time for time, _ in even_periods])
    even_values = np.concatenate([values for _, values in even_periods])

    return (odd_time, odd_values), (even_time, even_values)


def remove_time_series_noise(
    data: tuple[tuple[IterableOfSeries, IterableOfSeries], Iterable[tuple[Id, PeriodicEvent]]],
    gap: float,
    spline: Spline,
    include_empty_segments: bool,
) -> TimeWithSeries:
    """Remove time series low-frequency variablity.

    Args:
        data (tuple[ tuple[IterableOfSeries, IterableOfSeries], Iterable[tuple[Id,
        PeriodicEvent]]]): Time, values, etc
        gap (floating): Minimum width of the gap in time units at which the time series is divided
            into smaller parts
        spline (Spline): A spline object to use in time series fitting

    Returns:
        TimeWithSeries: Time and flattened values
    """
    (time, values), events_with_tce_id = data
    events = [event for _, event in events_with_tce_id]
    flattened_values = flatten_time_series(
        time,
        values,
        events,
        gap,
        spline,
        include_empty_segments,
    )
    return np.concatenate(list(time)), flattened_values


def read_data(
    cfg: CreateFeaturesConfig, spark: SparkSession
) -> tuple[RDD[TTce], RDD[TStellarParameters], RDD[TTimeSeries]]:
    """Read and map TCE, stellar parameters and time series.

    Args:
        spark_cfg (SparkPreprocessingConfig): Spark config with mappers and data sources
        spark (SparkSession): Spark session
        format (DataFormat): Data input format
        cache (bool, optional): Whether to cache data to improve performance. Defaults to False.

    Returns:
        tuple[RDD[TTce], RDD[TStellarParameters], RDD[TTimeSeries]]: Mapped TCEs, stellar
        parameters and time series
    """
    tce_mapper: Mapper[TTce] = hydra.utils.instantiate(cfg.tce_mapper)
    stellar_params_mapper: Mapper[TStellarParameters] = hydra.utils.instantiate(
        cfg.stellar_params_mapper
    )
    time_series_mapper: Mapper[TTimeSeries] = hydra.utils.instantiate(cfg.time_series_mapper)

    tces = read_objects(spark, cfg.tce_source, tce_mapper, cfg.input_format)
    stellar_params = read_objects(
        spark, cfg.stellar_params_source, stellar_params_mapper, cfg.input_format
    )    

    time_series = read_objects(spark, cfg.time_series_source, time_series_mapper, cfg.input_format)

    if cfg.spark.cache_input_data:
        return tces.cache(), stellar_params.cache(), time_series.cache()

    return tces, stellar_params, time_series


NormalizationData: TypeAlias = tuple[Id, Series, float, float]


def compute_view_normalization_data(data: Iterable[tuple[Id, Series]]) -> list[NormalizationData]:
    """Compute view median and absolute value of the minimum of the view.

    Args:
        data (Iterable[tuple[Id, Series]]): Time series views with TCE ID

    Returns:
        list[NormalizationData]: TCE ID, view, median(view), abs(min(view)) for each view in `data`
    """
    # TCE ID, view, median(view), abs(min(view))
    normalization_data: list[NormalizationData] = []

    for tce_id, view in data:
        view_median = np.nanmedian(view)
        view -= view_median

        view_abs_min = np.abs(np.nanmin(view))
        view /= view_abs_min

        normalization_data.append((tce_id, view, view_median, view_abs_min))

    return normalization_data


def compute_centroid_shift(data: tuple[IterableOfSeries, IterableOfSeries]) -> ListOfSeries:
    """Calculate the centroid shift as the Euclidean distance of the x and y coordinates.

    Args:
        data (tuple[IterableOfSeries, IterableOfSeries]): X and Y centroid coordinates in time

    Returns:
        ListOfSeries: Euclidean distance of the x and y coordinates of the centroid
    """
    xs, ys = data
    return [compute_euclidean_distance(np.vstack([x, y])) for x, y in zip(xs, ys)]


def create_secondary_time_series(
    data: tuple[TimeWithSeries, Iterable[tuple[Id, PeriodicEvent]]],
) -> TimeWithSeries:
    """Create a time series with only the secondary transits preserved.

    Args:
        data (tuple[TimeWithSeries, Iterable[tuple[Id, PeriodicEvent]]]): Time, values,
        iterable of TCE ID value and secondary transit event

    Returns:
        TimeWithSeries: Time series with non-secondary transits removed
    """
    (time, values), events = data
    secondary_time = []
    secondary_values = []

    for _, event in events:
        masked_time, masked_values = remove_events(
            [time],
            [event],
            [values],
            SecondaryTransitAdjustedPadding(event.secondary_phase),
        )
        secondary_time.append(np.concatenate(masked_time))
        secondary_values.append(np.concatenate(masked_values))

    return np.concatenate(secondary_time), np.concatenate(secondary_values)


def normalize_view_median_abs_min(
    data: tuple[Iterable[tuple[Id, Series]], Iterable[tuple[Id, float, float]]],
) -> list[tuple[Id, Series]]:
    """Normalize time series views using view median and absolute value of the minimum of the view.

    Normalization: `view -= median(view); view /= max(abs(min(view)), min(view))`.
    This normalization ensures that the median view is 0 and the minimum value is -1.

    Args:
        data (tuple[Iterable[tuple[Id, Series]], Iterable[tuple[Id, float, float]]]): Iterable of
        (TCE ID, series) and iterable of (TCE ID, median(view), abs(min(view)))

    Returns:
        list[tuple[Id, Series]]: Normalized views with a median of 0 and a minimum of -1
    """
    views, norm_data = data
    sorted_views = sorted(views, key=lambda e: e[0])
    sorted_norm_data = sorted(norm_data, key=lambda e: e[0])
    normalized_views: list[tuple[Id, Series]] = []

    for (tce_id, view), (_, median, abs_min) in zip(sorted_views, sorted_norm_data):
        view -= median
        view /= max(abs_min, np.min(view))
        normalized_views.append((tce_id, view))

    return normalized_views


def normalize_view_median_std(view: Series, std: float) -> Series:
    """Normalize the time series view using the view median and global standard deviation.

    Normalization: `view -= median(view); view /= std`.
    The standard deviation `std` should be calculated across all training views.

    Args:
        view (Series): Time series view
        std (float): Global (calculated for all training views) standard deviation

    Returns:
        Series: A normalized view of time series
    """
    view -= np.nanmedian(view)
    view /= std
    return view


T = TypeVar("T")


def train_test_val_split_rdd(
    data: RDD[tuple[Hashable, ...]],
    *,
    train_keys: Iterable[Hashable],
    test_keys: Iterable[Hashable],
    val_keys: Iterable[Hashable],
) -> tuple[RDD[tuple[Hashable, ...]], RDD[tuple[Hashable, ...]], RDD[tuple[Hashable, ...]]]:
    """Split keyed RDD into training, test and validation RDDs.

    Args:
        data (RDD[tuple[Hashable, ...]]): Data to split. Must be in form of key-value tuples
        train_keys (Iterable[Hashable]): Keys of training examples
        test_keys (Iterable[Hashable]): Keys of test examples
        val_keys (Iterable[Hashable]): Keys of validation examples

    Returns:
        tuple[RDD[tuple[Hashable, ...]], RDD[tuple[Hashable, ...]], RDD[tuple[Hashable, ...]]]:
        Train, test and validation RDDs
    """
    train = data.filter(lambda d: d[0] in train_keys)
    test = data.filter(lambda d: d[0] in test_keys)
    val = data.filter(lambda d: d[0] in val_keys)
    return train, test, val


def compute_views_std(views: RDD[Series]) -> float:
    """Calculate the standard deviation of all time series views.

    Args:
        views (RDD[Series]): Time series views

    Returns:
        float: Standard deviation of all time series views
    """
    return views.aggregate(
        np.array([]),  # Initialize with an empty array
        lambda a, b: np.append(a, b),  # Append views togather
        lambda a, b: np.concatenate([a, b]),  # Concatenate views from different partitions
    ).std()


def create_secondary_events(tces: RDD[TCE]) -> RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]]:
    """Create RDD of TCE secondary trainst events.

    Args:
        tces (RDD[TCE]): TCEs

    Returns:
        RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]]: Secondary trainst events keyed by TCE ID
    """
    return tces.map(
        lambda tce: (
            tce.target_id,
            (
                tce.id,
                PeriodicEvent(
                    tce.event.epoch + tce.event.secondary_phase,
                    tce.event.duration,
                    tce.event.period,
                    secondary_phase=None,
                ),
            ),
        ),
    ).groupByKey()


def create_views_and_normalized_data(
    create_view_func: GenerateViewFunction,
    events: RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]],
    series: RDD[tuple[Id, TimeWithSeries]],
) -> tuple[RDD[tuple[tuple[Id, Id], Series]], RDD[tuple[Id, list[tuple[Id, float, float]]]]]:
    """Create time series and compute its median and absolute value of the minimum of the view.

    Args:
        create_view_func (GenerateViewFunction): Function to generate view
        events (RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]]): TCE transit events keyed by
        target id and TCE id e.g.: `(target_id1, [(tce_id1, event1), (tce_id2, event2)])`
        series (RDD[tuple[Id, TimeWithSeries]]): Time series keyed by target id

    Returns:
        tuple[RDD[tuple[tuple[Id, Id], Series]], RDD[tuple[Id, list[tuple[Id, float, float]]]]]:
        Views keyd by target id and TCE id with `median(view)` and `abs(min(view))`
    """
    normalization_data = (
        series.join(events).mapValues(create_view_func).mapValues(compute_view_normalization_data)
    )

    # data[0] = target id
    views = normalization_data.mapValues(
        lambda data: [(tce_id, view) for tce_id, view, _, _ in data],
    ).flatMap(lambda data: [((data[0], tce_id), view) for tce_id, view in data[1]])

    median_abs_min = normalization_data.mapValues(
        lambda data: [(tce_id, median, abs_min) for tce_id, _, median, abs_min in data],
    )
    return views, median_abs_min


def join_final_data(
    tces: RDD[tuple[tuple[Id, Id], dict[str, Any]]],
    stellar_params: RDD[tuple[Id, dict[str, Any]]],
    views: RDD[tuple[tuple[Id, Id], Series]],
    view_names: Iterable[str],
) -> RDD[dict[str, str | list[float] | int | float]]:
    """Join TCEs, stellar parameters and time series views for each TCE into a single dataset.

    Args:
        tces (RDD[tuple[tuple[Id, Id], dict[str, Any]]]): Final TCE scalar data
        stellar_params (RDD[tuple[Id, dict[str, Any]]]): Final stellar parameters
        views (RDD[tuple[tuple[Id, Id], Series]]): Final time-series views
        view_names (Iterable[str]): Time series view names used as dictionary keys

    Returns:
        RDD[dict[str, str | list[float] | int | float]]: Joined data in form of dictionaries
    """
    final_data = (
        views.mapValues(
            lambda views: [v.data[0] for v in views],
        )  # Cast pyspark.ResultIterable back to numpy array
        .join(tces)
        .mapValues(lambda d: (d[1]["target_id"], d))  # Create key-value pairs (target_id as key)
        .values()
        .join(stellar_params)
        .mapValues(lambda d: (d[0][0], d[0][1], d[1]))  # map to ([views], tce, stellar_params)
        .mapValues(
            lambda d: {"id": f"{d[1]['target_id']}-{d[1]['id']}"}
            | {
                k: v.tolist() for k, v in zip(view_names, d[0])
            }  # noqa cast numpy array to python list for spark schema infering
            | {
                f"tce_{k}": v for k, v in d[1].items()
            }  # noqa add 'tce' prefix to distinguise from stellar params
            | {
                f"stellar_{k}": v for k, v in d[2].items()
            },  # noqa add 'stellar' prefix to distinguise from tce
        )
        .cache()
    )

    return final_data


def create_secondary_views(
    generate_view_func: GenerateViewFunction,
    events: RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]],
    secondary_events: RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]],
    series: RDD[tuple[Id, TimeWithSeries]],
    median_abs_min: RDD[tuple[Id, list[tuple[Id, float, float]]]],
) -> RDD[tuple[tuple[Id, Id], Series]]:
    """Create views of secondary TCE transits.

    Args:
        generate_view_func (GenerateViewFunction): Function that generates a time series view
        events (RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]]]): TCE transit events
        secondary_events (RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]]]: TCE secondary
        transit events
        series (RDD[tuple[Id, TimeWithSeries]]): Time series
        median_abs_min (RDD[tuple[Id, list[tuple[Id, float, float]]]]]: The median and absolute
        value of the view minimum needed to normalize the views

    Returns:
        RDD[tuple[tuple[Id, Id], Series]]: Normalized views of secondary TCE transit events keyed
        by (target id, TCE id)
    """
    return (
        series.join(events)
        .mapValues(create_secondary_time_series)
        .join(secondary_events)
        .mapValues(generate_view_func)
        .join(median_abs_min)
        .mapValues(normalize_view_median_abs_min)
        .flatMap(
            lambda data: [((data[0], tce_id), view) for tce_id, view in data[1]],
        )  # Keyed by (target_id, tce_id)
    )


def create_odd_even_views(
    generate_view_func: GenerateViewFunction,
    events: RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]],
    series: RDD[tuple[Id, TimeWithSeries]],
    median_abs_min: RDD[tuple[Id, list[tuple[Id, float, float]]]],
) -> tuple[RDD[tuple[tuple[Id, Id], Series]], RDD[tuple[tuple[Id, Id], Series]]]:
    """Create time series views of odd and even orbital periods.

    Args:
        generate_view_func (GenerateViewFunction): Function that generates a time series view
        events (RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]]): TCE transit events
        series (RDD[tuple[Id, TimeWithSeries]]): Time series
        median_abs_min (RDD[tuple[Id, list[tuple[Id, float, float]]]]): The median and absolute
        value of the view minimum needed to normalize the views

    Returns:
        tuple[RDD[tuple[tuple[Id, Id], Series]], RDD[tuple[tuple[Id, Id], Series]]]: Normalized odd
        and even time series views keyed by (target id, TCE id)
    """
    odd_even_flux = series.join(events).mapValues(split_periods_to_odd_even)
    odd_flux = odd_even_flux.mapValues(lambda data: data[0])
    even_flux = odd_even_flux.mapValues(lambda data: data[1])

    def _create_view(
        flux: RDD[tuple[Id, TimeWithSeries]],
        events: RDD[tuple[Id, Iterable[tuple[Id, PeriodicEvent]]]],
        generate_view_func: GenerateViewFunction,
        median_abs_min: RDD[tuple[Id, list[tuple[Id, float, float]]]],
    ) -> RDD[tuple[tuple[Id, Id], Series]]:
        return (
            flux.join(events)
            .mapValues(generate_view_func)
            .join(median_abs_min)
            .mapValues(normalize_view_median_abs_min)
            .flatMap(
                lambda data: [((data[0], tce_id), view) for tce_id, view in data[1]],
            )  # Keyed by (target_id, tce_id)
        )

    odd_flux_views = _create_view(odd_flux, events, generate_view_func, median_abs_min)
    even_flux_views = _create_view(even_flux, events, generate_view_func, median_abs_min)
    return odd_flux_views, even_flux_views


def normalize_scalar(
    spark: SparkSession,
    rdd: RDD[Any],
    train_ids: Iterable[Id],
    id_column: str,
    excluded_columns: Iterable[str],
    sigma_cut: float,
) -> RDD[dict[Hashable, Any]]:
    """Normalize scalar values.

    Normalization: `(rdd - median(train_rdd)) / std(train_rdd)`

    This function CHANGES `rdd` data type to `dict[Hashable, Any]]`!!!

    Args:
        spark (SparkSession): Spark session
        rdd (RDD[Any]): Scalar data to normalize
        train_ids (Iterable[Id]): Traning ids
        id_column (str): Id column name
        excluded_columns (Iterable[str]): Columns to exclude from normalization
        sigma_cut (float): Clip values beyond `sigma_cut * std(rdd)`

    Returns:
        RDD[dict[Hashable, Any]]: Normalized `rdd`
    """
    df = pd.DataFrame.from_records(map(map_dataclass_to_flat_dict, rdd.collect()))
    train_df = df.query(f"{id_column} in @train_ids")
    normalized_data = normalize_data_frame_median_std(
        df, train_df, excluded_columns, sigma_cut
    ).to_dict("records")
    return spark.sparkContext.parallelize(normalized_data)


def rename_dataproc_spark_files(
    client: storage.Client,
    bucket_or_name: str | storage.Bucket,
    source_dir: str,
    format: str,
    partitioned_by: Sized,
    delete_crc_files: bool = False,
    batch_size: int = 100,
) -> None:
    """Rename partitioned files stored on GCS. Use partitioning columns values as filenames.

    Args:
        client (storage.Client): Google Cloud Storage (GCS) client
        bucket_or_name (str | storage.Bucket): Bucket or name with partitioned files saved
        source_dir (str): Folder (prefix in GCS nomenclature) with partitioned files
        format (str): Data format
        partitioned_by (Sized): Names of partitioning columns
        delete_crc_files (bool, optional): Whether to delete CRC files. Defaults to False.
        batch_size (int, optional): REST request batch size. Can't be greater than 100.
        Defaults to 100.

    Raises:
        ValueError: `batch_size` < 0 or `batch_size` > 100
    """
    if not (0 < batch_size <= 100):
        raise ValueError(f"'batch_size' must be > 0 and <= 100, but got {batch_size=}")

    bucket = client.get_bucket(bucket_or_name)
    files_pattern = "*/*".join(partitioned_by) if partitioned_by else ""

    if files_pattern:
        files_pattern = f"*{files_pattern}*/"

    blobs: list[storage.Blob] = list(
        bucket.list_blobs(prefix=source_dir, match_glob=f"**/{files_pattern}*part*.{format}"),
    )
    blob_batches = get_chunks(blobs, batch_size)

    with ProgressBar() as bar:
        rename_task_id = bar.add_task("Renaming GCP files", total=len(blobs))

        for batch in blob_batches:
            for blob in batch:
                with client.batch():
                    _, _, blob_extension = blob.name.rpartition(".")
                    ids_pattern = ".*".join(
                        [f"((?<={partition}=)\d*)" for partition in partitioned_by],
                    )  # `\d` jest tylko dla [0-9] a co z str, guid, itp?
                    ids_parts = list(range(1, len(partitioned_by) + 1))
                    ids = re.search(ids_pattern, blob.name).group(*ids_parts)
                    ids_string = "-".join(ids)
                    new_name = f"{source_dir}/{ids_string}.{blob_extension}"
                    bucket.copy_blob(blob, bucket, new_name)  # rename partitioned files

            bar.advance(rename_task_id, batch_size)

    # delete partitioned files
    bucket.delete_blobs(blobs)

    if delete_crc_files:
        bucket.delete_blobs(list(bucket.list_blobs(prefix=source_dir, match_glob="**/*.crc")))


def init_spark_session(local: bool, cfg: SparkConfig) -> SparkSession:
    """Initiate a `SparkSession`, optionally setting configuration properties.

    In local mode, use all available CPU cores.

    Args:
        cfg (CreateFeaturesConfig): Script configuration

    Returns:
        SparkSession: Spark session
    """
    builder: SparkSession.Builder = (
        SparkSession.builder if local else SparkSession.builder.master("local[*]")
    )

    if cfg.config_properties:
        for key, value in cfg.config_properties.items():
            builder = builder.config(key, value)

    return builder.appName(cfg.app_name).getOrCreate()


def save_rdd(
    data: RDD[Any],
    output: str,
    output_format: Literal["json", "parquet", "tfrecord"],
    train_ids: Iterable[Id],
    test_ids: Iterable[Id],
    val_ids: Iterable[Id],
    partition_by: list[str] | None = None,
    schema: StructType | None = None,
) -> None:
    """Split the RDD into test, training, and validation RDDs and save them in separate locations.

    The split RDDs will be saved to `{output}/[test|train|validation]` locations.

    Args:
        data (RDD[Any]): Data to save
        output (str): Output location
        output_format (Literal['json', 'parquet', 'tfrecord']): Output format
        train_ids (Iterable[Id]): Trainging target ids
        test_ids (Iterable[Id]): Test target ids
        val_ids (Iterable[Id]): Validation target ids
        partition_by (list[str] | None, optional): Names of partitioning columns. Defaults to None.
        schema (StructType | None, optional): Schema of RDD elements. Will be inferred if None.
        Inference is time-consuming; use schema if possible. Defaults to None.
    """
    final_partitioned_data = train_test_val_split_rdd(
        data,
        train_keys=train_ids,
        test_keys=test_ids,
        val_keys=val_ids,
    )
    final_data_outputs = (f"{output}/train", f"{output}/test", f"{output}/validation")

    for data, output_dir in zip(final_partitioned_data, final_data_outputs):
        writer = data.values().toDF(schema).write

        if output_format == "tfrecord":
            writer = writer.option("recordType", "Example")

        writer.save(output_dir, format=output_format, mode="overwrite", partitionBy=partition_by)


def dict_key_by(data: dict[Hashable, Any], keys: list[str]) -> Any | tuple[Any, ...]:
    """Generate a tuple or single value by extracting value(s) from `data` using `keys`.

    Args:
        data (dict[Hashable, Any]): Data to extratc key values from
        keys (list[str]): Names of `data` columns to extract key values

    Raises:
        ValueError: Empty `keys`

    Returns:
        Any | tuple[Any, ...]: Data value or values extracted based on `keys`
    """
    if not keys:
        raise ValueError("Cannot key by using empty list of keys")

    if len(keys) == 1:
        return data[keys[0]]

    return tuple(data[col] for col in keys)
