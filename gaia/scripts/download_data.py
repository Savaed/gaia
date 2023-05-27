import asyncio
from textwrap import dedent

import duckdb
import hydra
from omegaconf import OmegaConf

from gaia.asynchronous import prepare_loop
from gaia.config import Config
from gaia.data.models import Id
from gaia.downloaders import KeplerDownloader, RestDownloader
from gaia.enums import Cadence
from gaia.io import FileSaver
from gaia.log import logger
from gaia.ui.cli import print_header


SCRIPT_DESCRIPTION = """
    This script downloads star or system brightness time series and scalar stellar and TCE data and
    saves them in unaltered, raw form at user-specified locations.

    In order to download time series, a file with target identifiers is required, e.g. a file with
    TCE data.

    The script has a mechanism for retrying the download in case of an error.

    The metadata of the time series downloaded once is saved in the file and time series are not
    downloaded again the next time the script is run (unless the metadata file has been deleted).

    Note that the size of the downloaded data is significant (hundreds of GB).

    """


def get_target_ids(cfg: Config) -> list[Id]:
    target_ids = duckdb.sql(
        f"SELECT DISTINCT {cfg.data.load.target_id_column} FROM '{cfg.data.load.tce_file}';",
    ).fetchall()
    return [id[0] for id in target_ids]


def print_downloading_header(cfg: Config, description: str) -> None:
    if cfg.mission.lower() == "kepler":
        description += (
            "🚩 There is one table (Certified False Positive, [italic]CFP[/italic]) that cannot be "
            "downloaded through this script. Please download it manually from "
            "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=fpwg\n"
        )

    print_header(description)


def create_downloader(cfg: Config) -> RestDownloader:
    saver = FileSaver(
        tables_dir=cfg.data.raw_tables_dir.as_posix(),
        time_series_dir=cfg.data.raw_time_series_dir.as_posix(),
    )
    downloader = KeplerDownloader(
        saver=saver,
        cadence=Cadence(cfg.data.load.observation_cadence),
        nasa_base_url=cfg.data.load.api.nasa_base_url,
        mast_base_url=cfg.data.load.api.mast_base_url,
    )
    return downloader


async def main(cfg: Config) -> int:
    prepare_loop(asyncio.get_running_loop())

    cfg_dict = OmegaConf.to_object(cfg)
    print_downloading_header(cfg, dedent(SCRIPT_DESCRIPTION))
    cfg = Config(**cfg_dict)  # Resolve configs via hydra and validate via pydantic

    print(cfg.data.raw_tables_dir.absolute())

    if not cfg.data.load.verbose:
        logger.disable("gaia.downloaders")

    downloader = create_downloader(cfg)
    try:
        if not cfg.data.load.skip_tables:
            await downloader.download_tables(cfg.data.load.api.requests)  # type: ignore

        await downloader.download_time_series(get_target_ids(cfg))
    except:  # noqa
        logger.info("Script shutdown")

    return 0


@logger.catch(message="Unexpected error occurred")
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main_wrapper(cfg: Config) -> int:
    raise SystemExit(asyncio.run(main(cfg)))


if __name__ == "__main__":
    main_wrapper()
