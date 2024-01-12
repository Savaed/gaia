import asyncio
from pathlib import Path

import duckdb
import hydra
from omegaconf import OmegaConf

from gaia.asynchronous import prepare_loop
from gaia.config import AppConfig
from gaia.data.models import Id
from gaia.downloaders import RestDownloader
from gaia.log import logger
from gaia.ui.cli import print_header


def get_target_ids(target_id_column: str, tce_file_path: Path) -> list[Id]:
    target_ids = duckdb.sql(
        f"SELECT DISTINCT {target_id_column} FROM '{tce_file_path}';",
    ).fetchall()
    return [id[0] for id in target_ids]


async def main(cfg: AppConfig) -> int:
    cfg = AppConfig(**OmegaConf.to_object(cfg))

    prepare_loop(asyncio.get_running_loop())
    print_header(cfg.download.script_description)

    downloader: RestDownloader = hydra.utils.instantiate(cfg.download.downloader)
    try:
        # if cfg.download.download_tables:
        #     await downloader.download_tables(cfg.download.tables)
        if cfg.download.download_time_series:
            ids = get_target_ids(cfg.download.tce_target_id_column, cfg.download.tce_filepath)
            await downloader.download_time_series(ids)
    except Exception as ex:
        logger.info(f"Script shutdown. {ex or ''}")

    return 0


@logger.catch(message="Unexpected error occurred")
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main_wrapper(cfg: AppConfig) -> int:
    return asyncio.run(main(cfg))


if __name__ == "__main__":
    raise SystemExit(main_wrapper())
