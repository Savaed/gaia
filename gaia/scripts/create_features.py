import os
from functools import partial
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from gaia.config import AppConfig
from gaia.data.models import change_tce_label_to_int
from gaia.log import logger
from gaia.model_selection import train_test_val_split
from gaia.spark import (
    FlattenFunction,
    GenerateViewFunction,
    compute_centroid_shift,
    compute_views_std,
    create_odd_even_views,
    create_secondary_events,
    create_secondary_views,
    create_views_and_normalized_data,
    dict_key_by,
    init_spark_session,
    join_final_data,
    normalize_scalar,
    normalize_view_median_std,
    read_data,
    save_rdd,
)


# Output data fields. MUST be the same as 'view_names' variable from this file, KepplerTCE
# (plus 'tce' prefix) and KeplerStellarParameters (plus 'stellar' prefix) and 'id' string
FIELDS = [
    StructField("even_flux", ArrayType(DoubleType(), False)),
    StructField("global_centroid_shift", ArrayType(DoubleType(), False)),
    StructField("global_flux", ArrayType(DoubleType(), False)),
    StructField("local_centroid_shift", ArrayType(DoubleType(), False)),
    StructField("local_flux", ArrayType(DoubleType(), False)),
    StructField("odd_flux", ArrayType(DoubleType(), False)),
    StructField("secondary_flux", ArrayType(DoubleType(), False)),
    StructField("stellar_density", DoubleType()),
    StructField("stellar_effective_temperature", DoubleType()),
    StructField("stellar_id", LongType()),
    StructField("stellar_mass", DoubleType()),
    StructField("stellar_metallicity", DoubleType()),
    StructField("stellar_radius", DoubleType()),
    StructField("stellar_surface_gravity", DoubleType()),
    StructField("tce_bootstrap_false_alarm_proba", DoubleType()),
    StructField("tce_duration", DoubleType()),
    StructField("tce_epoch", DoubleType()),
    StructField("tce_fitted_period", DoubleType()),
    StructField("tce_id", LongType()),
    StructField("tce_label", IntegerType()),
    StructField("tce_name", StringType()),
    StructField("tce_opt_ghost_core_aperture_corr", DoubleType()),
    StructField("tce_opt_ghost_halo_aperture_corr", DoubleType()),
    StructField("tce_period", DoubleType()),
    StructField("tce_radius", DoubleType()),
    StructField("tce_rolling_band_fgt", DoubleType()),
    StructField("tce_secondary_phase", DoubleType()),
    StructField("tce_secondary_transit_depth", DoubleType()),
    StructField("tce_target_id", LongType()),
    StructField("tce_transit_depth", DoubleType()),
    StructField("id", StringType()),
]

GCP_CONFIG_ARCHIVE_NAME = "configs.zip"


if os.getenv("DATAPROC_SERVERLESS_ENVIRONMENT"):
    # Run this only on GCP Dataproc.
    Path(GCP_CONFIG_ARCHIVE_NAME).rename("configs")
    CONFIG_DIR = "configs"
else:
    CONFIG_DIR = "../../configs"


@logger.catch(message="Unexpected error occurred")
@hydra.main(CONFIG_DIR, "config", "1.3")
def main(cfg: AppConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    cfg = AppConfig(**OmegaConf.to_object(cfg))
    script_cfg = cfg.create_features

    # Initialization
    local_mode = not script_cfg.dataproc
    spark = init_spark_session(local=local_mode, cfg=script_cfg.spark)

    flatten_time_series_func: FlattenFunction = hydra.utils.instantiate(script_cfg.flatten_func)
    generate_local_view_func_spark: GenerateViewFunction = hydra.utils.instantiate(
        script_cfg.generate_local_view_func,
    )
    generate_global_view_func_spark: GenerateViewFunction = hydra.utils.instantiate(
        script_cfg.generate_global_view_func,
    )

    # Read data and split into test, train and validation sets
    tces, stellar_params, time_series = read_data(script_cfg, spark)
    train_ids, test_ids, val_ids = train_test_val_split(
        tces.map(lambda tce: tce.target_id).distinct().collect(),
        test_size=script_cfg.test_size,
        validation_size=script_cfg.validation_size,
        seed=script_cfg.seed,
    )

    # Normalize TCEs
    normalized_tces = (
        normalize_scalar(
            spark,
            tces,
            train_ids,
            id_column=script_cfg.tce_target_id_column,
            excluded_columns=script_cfg.tce_excluded_columns,
            sigma_cut=script_cfg.sigma_cut,
        )
        .map(change_tce_label_to_int)
        .keyBy(partial(dict_key_by, keys=script_cfg.tce_key_by))
    )

    # Normalize stellar parameters
    normalized_stellar_params = normalize_scalar(
        spark,
        stellar_params,
        train_ids,
        id_column=script_cfg.stellar_params_target_id_column,
        excluded_columns=script_cfg.stellar_params_excluded_columns,
        sigma_cut=script_cfg.sigma_cut,
    ).keyBy(partial(dict_key_by, keys=script_cfg.stellar_params_key_by))

    # Extract primary and secondary events from TCEs
    events = tces.map(lambda tce: (tce.target_id, (tce.id, tce.event))).groupByKey()
    secondary_events = create_secondary_events(tces)

    # Group time series observation periods for targets (flux)
    time = time_series.map(
        lambda t: (t[script_cfg.time_series_id_key], t[script_cfg.time_key])
    ).groupByKey()
    flux = time_series.map(
        lambda t: (t[script_cfg.time_series_id_key], t[script_cfg.flux_key])
    ).groupByKey()

    # Group time series observation periods for targets (centroid_x, centroid_y)
    centroid_y = time_series.map(
        lambda t: (t[script_cfg.time_series_id_key], t[script_cfg.centroid_y_key]),
    ).groupByKey()
    centroid_x = time_series.map(
        lambda t: (t[script_cfg.time_series_id_key], t[script_cfg.centroid_x_key]),
    ).groupByKey()

    centroid_shift = centroid_x.join(centroid_y).mapValues(compute_centroid_shift)

    # Remove centroid shift noise
    flattened_centroid_shif = (
        time.join(centroid_shift).join(events).mapValues(flatten_time_series_func)
    )

    # Remove flux noise
    flattened_flux = time.join(flux).join(events).mapValues(flatten_time_series_func)

    # Generate local and global views (flux, centriod shift)
    local_flux_normalized_views, local_flux_median_abs_min = create_views_and_normalized_data(
        generate_local_view_func_spark,
        events,
        flattened_flux,
    )
    global_flux_normalized_views, _ = create_views_and_normalized_data(
        generate_global_view_func_spark,
        events,
        flattened_flux,
    )
    local_centroid_shift_views, _ = create_views_and_normalized_data(
        generate_local_view_func_spark,
        events,
        flattened_centroid_shif,
    )
    global_centroid_shift_views, _ = create_views_and_normalized_data(
        generate_global_view_func_spark,
        events,
        flattened_centroid_shif,
    )

    # Compute global train centroid shift std
    local_centroid_shift_views_train = local_centroid_shift_views.filter(
        lambda data: data[0][1] in train_ids
    )
    global_centroid_shift_views_train = global_centroid_shift_views.filter(
        lambda data: data[0][1] in train_ids
    )
    local_centroid_shift_views_std = compute_views_std(local_centroid_shift_views_train.values())
    global_centroid_shift_views_std = compute_views_std(global_centroid_shift_views_train.values())

    # Normalize centroid shift views
    local_normalize_centroid_shift_views_func = partial(
        normalize_view_median_std,
        std=local_centroid_shift_views_std,
    )
    global_normalize_centroid_shift_views_func = partial(
        normalize_view_median_std,
        std=global_centroid_shift_views_std,
    )
    global_centroid_shift_normalized_views = global_centroid_shift_views.mapValues(
        global_normalize_centroid_shift_views_func
    )
    local_centroid_shift_normalized_views = local_centroid_shift_views.mapValues(
        local_normalize_centroid_shift_views_func
    )

    odd_flux_normalized_views, even_flux_normalized_views = create_odd_even_views(
        generate_local_view_func_spark,
        events,
        flattened_flux,
        local_flux_median_abs_min,
    )

    secondary_flux_normalized_views = create_secondary_views(
        generate_local_view_func_spark,
        events,
        secondary_events,
        flattened_flux,
        local_flux_median_abs_min,
    )

    # Groups all views
    final_views = local_flux_normalized_views.groupWith(
        global_flux_normalized_views,
        local_centroid_shift_normalized_views,
        global_centroid_shift_normalized_views,
        odd_flux_normalized_views,
        even_flux_normalized_views,
        secondary_flux_normalized_views,
    )

    view_names = (
        "local_flux",
        "global_flux",
        "local_centroid_shift",
        "global_centroid_shift",
        "odd_flux",
        "even_flux",
        "secondary_flux",
    )

    final_data = join_final_data(
        normalized_tces,
        normalized_stellar_params,
        final_views,
        view_names,
    )

    save_rdd(
        data=final_data,
        output=script_cfg.output,
        output_format=script_cfg.output_format,
        train_ids=train_ids,
        test_ids=test_ids,
        val_ids=val_ids,
        partition_by=script_cfg.partition_by,
        schema=StructType(FIELDS),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
