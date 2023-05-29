import asyncio
import re
from pathlib import Path
from textwrap import dedent

import duckdb
import hydra
from omegaconf import OmegaConf

from gaia.asynchronous import prepare_loop
from gaia.config import Config
from gaia.data.converters import (
    CsvConverter,
    FitsConverter,
    FitsConvertingOutputFormat,
    FitsConvertingSettings,
)
from gaia.log import logger
from gaia.ui.cli import print_header


SCRIPT_DESCRIPTION = """
    Merge TCE data with other scalar data (e.g. KOI or CFP tables) to obtain TCE labels and names.
    Convert raw files to another format, optionally loading only part of the data or metadata.

    Metadata of once converted time series are saved in the file and the time series are not
    converted again the next time the script is run (unless the metadata file has been deleted).
"""


def merge_tce_file(
    select_sql: str,
    join_sql: str,
    label_conditions: list[str],
    output: Path,
) -> None:
    logger.info("Merging and labeling TCEs")
    case_when_sql = " ".join(label_conditions)

    match output.suffix:
        case ".csv":
            copy_args = "WITH (HEADER 1);"
        case ".parquet":
            copy_args = "(COMPRESSION ZSTD);"
        case _:
            copy_args = ";"

    label_column = "label_text"
    tce_label_sql = f"CASE {case_when_sql} END AS '{label_column}'"
    merge_sql = f"COPY ({select_sql}, {tce_label_sql} FROM {join_sql}) TO '{output}' {copy_args}"
    duckdb.execute(merge_sql)
    duckdb.execute(f"SELECT * FROM '{output}' LIMIT 1;")
    logger.bind(label_column=label_column, output_file=output.as_posix()).info(
        "TCEs merged and labeled",
    )


async def main(cfg: Config) -> int:
    prepare_loop(asyncio.get_running_loop())
    print_header(dedent(SCRIPT_DESCRIPTION))

    cfg = Config(**OmegaConf.to_object(cfg))
    cfg.data.interim_tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.interim_time_series_dir.mkdir(parents=True, exist_ok=True)

    merge_tce_file(
        cfg.data.tce_merge.select_sql,
        cfg.data.tce_merge.join_sql,
        cfg.data.tce_merge.label_conditions_case_sql,
        cfg.data.tce_merge.output_file,
    )

    CsvConverter().convert(
        inputs=cfg.data.convert.tce.inputs,
        output=cfg.data.convert.tce.output,
        include_columns=cfg.data.convert.tce.columns,
        columns_mapping=cfg.data.convert.tce.columns_map,
    )

    CsvConverter().convert(
        inputs=cfg.data.convert.stellar.inputs,
        output=cfg.data.convert.stellar.output,
        include_columns=cfg.data.convert.stellar.columns,
        columns_mapping=cfg.data.convert.stellar.columns_map,
    )

    fits_converter_settings = FitsConvertingSettings(
        data_header=cfg.data.convert.time_series.data_header,
        data_columns=cfg.data.convert.time_series.data_columns,
        meta_columns=cfg.data.convert.time_series.meta_columns,
        names_map=cfg.data.convert.time_series.names_map,
        output_format=FitsConvertingOutputFormat.PARQUET,
    )

    fits_converter = FitsConverter(fits_converter_settings)
    try:
        await fits_converter.convert(
            inputs=cfg.data.convert.time_series.inputs,
            output_dir=cfg.data.convert.time_series.output,
            path_target_id_pattern=re.compile(cfg.data.convert.time_series.path_target_id_pattern),
        )
    except KeyboardInterrupt:
        logger.info("Script shutdown")

    return 0


@logger.catch(message="Unexpected error occurred")
@hydra.main("../../configs", "config", version_base="1.3")
def main_wrapper(cfg: Config) -> int:
    raise SystemExit(asyncio.run(main(cfg)))


if __name__ == "__main__":
    main_wrapper()
