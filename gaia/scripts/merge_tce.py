from pathlib import Path

import duckdb
import hydra
from omegaconf import OmegaConf

from gaia.config import AppConfig
from gaia.data.converters import DUCKDB_COPY_ARGS
from gaia.log import logger
from gaia.ui.cli import print_header


def merge_tce_file(
    select_sql: str,
    join_sql: str,
    label_conditions: list[str],
    output: Path,
    label_column: str,
) -> None:
    logger.info("Merging and labeling TCEs")
    case_when_sql = " ".join(label_conditions)
    copy_args = DUCKDB_COPY_ARGS.get(output.suffix, ";")
    tce_label_sql = f"CASE {case_when_sql} END AS '{label_column}'"
    merge_sql = f"COPY ({select_sql}, {tce_label_sql} FROM {join_sql}) TO '{output}' {copy_args}"
    duckdb.execute(merge_sql)
    duckdb.execute(f"SELECT * FROM '{output}' LIMIT 1;")
    logger.bind(label_column=label_column, output_file=output.as_posix()).info(
        "TCEs merged and labeled",
    )


@logger.catch(message="Unexpected error occurred")
@hydra.main("../../configs", "config", version_base=None)
def main(cfg: AppConfig) -> int:
    cfg = AppConfig(**OmegaConf.to_object(cfg))
    print_header(cfg.preprocessing.tce_merge.script_description)
    merge_tce_file(
        cfg.preprocessing.tce_merge.select_sql,
        cfg.preprocessing.tce_merge.join_sql,
        cfg.preprocessing.tce_merge.labels_case_sql,
        cfg.preprocessing.tce_merge.output,
        cfg.preprocessing.tce_merge.label_column,
    )
    return 0


if __name__ == "__main__":
    main()
