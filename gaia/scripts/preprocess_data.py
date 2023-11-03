import asyncio
from pathlib import Path
from typing import Any

import duckdb
import hydra
from omegaconf import OmegaConf

from gaia.asynchronous import prepare_loop
from gaia.config import AppConfig
from gaia.log import logger
from gaia.ui.cli import print_header


def remove_dict_fields(dct: dict[str, Any], *fields: str) -> dict[str, Any]:
    for field in fields:
        del dct[field]

    return dct


def merge_tce_file(
    select_sql: str,
    join_sql: str,
    label_conditions: list[str],
    output: Path,
    label_column: str,
) -> None:
    logger.debug("Merging and labeling TCEs")
    case_when_sql = " ".join(label_conditions)
    tce_label_sql = f"CASE {case_when_sql} END AS '{label_column}'"
    merge_sql = f"COPY ({select_sql}, {tce_label_sql} FROM {join_sql}) TO '{output}';"
    duckdb.execute(merge_sql)
    duckdb.execute(f"SELECT * FROM '{output}' LIMIT 1;")
    logger.bind(label_column=label_column, output_file=output.as_posix()).debug(
        "TCEs merged and labeled",
    )


async def main(cfg: AppConfig) -> int:
    cfg = AppConfig(**OmegaConf.to_object(cfg))
    preprocess_cfg = cfg.preprocess_data

    prepare_loop(asyncio.get_running_loop())
    print_header(preprocess_cfg.script_description)

    merge_tce_file(
        select_sql=preprocess_cfg.tce_merge.select_sql,
        join_sql=preprocess_cfg.tce_merge.join_sql,
        label_conditions=preprocess_cfg.tce_merge.labels_case_sql,
        output=preprocess_cfg.tce_merge.output,
        label_column=preprocess_cfg.tce_merge.label_column,
    )

    # HACK: Remove 'converter_type' field from converter config dictionary to use with
    # `hydra.utils.instantiate` as at this time hydra cannot ignore additional arguments when
    # instantiating '_target_' object
    field_to_remove = "converter_type"
    tce_converter = hydra.utils.instantiate(
        remove_dict_fields(preprocess_cfg.tce_conversion.converter.__dict__, field_to_remove),
    )
    staller_converter = hydra.utils.instantiate(
        remove_dict_fields(
            preprocess_cfg.stellar_params_conversion.converter.__dict__,
            field_to_remove,
        ),
    )
    fits_converter = hydra.utils.instantiate(
        remove_dict_fields(
            preprocess_cfg.time_series_conversion.converter.__dict__,
            field_to_remove,
        ),
    )

    tce_converter.convert(**preprocess_cfg.tce_conversion.args.model_dump())
    staller_converter.convert(**preprocess_cfg.stellar_params_conversion.args.model_dump())

    try:
        await fits_converter.convert(**preprocess_cfg.time_series_conversion.args.model_dump())
    except KeyboardInterrupt:
        logger.info("Script shutdown")

    return 0


@logger.catch(message="Unexpected error occurred")
@hydra.main("../../configs", "config", version_base="1.3")
def main_wrapper(cfg: AppConfig) -> int:
    return asyncio.run(main(cfg))


if __name__ == "__main__":
    raise SystemExit(main_wrapper())
