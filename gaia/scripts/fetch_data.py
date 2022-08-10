# pylint: skip-file

import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import asyncio
import logging
import os
from typing import Iterable

import hydra
import tensorflow as tf
from rich.status import Status
from tensorflow.python.framework.errors import NotFoundError

from gaia.api import InvalidRequestOrResponse, MastApi, NasaApi, get_mast_urls
from gaia.enums import Cadence
from gaia.hydra_config import BaseConfig
from gaia.io.kepler import gfile_write, pd_read_csv
from gaia.utils.asynchronous import cancel_tasks, prepare_loop
from gaia.utils.progress import ProgressBar


log = logging.getLogger(__name__)


def get_kepids(tce_filename: str) -> set[int]:
    return set(pd_read_csv(tce_filename)["kepid"])


def get_processed_mast_urls(filename: str) -> set[str]:
    try:
        log.debug("Fetching processed MAST URLs")
        with tf.io.gfile.GFile(filename) as gf:
            urls = gf.readlines()
            return set(map(str.strip, urls))
    except NotFoundError:
        return set()


def get_kepler_urls(
    base_url: str, kepids: Iterable[int], cadence: Cadence, processed_urls_filename: str
) -> set[str]:
    log.info("Determining MAST URLs")
    urls = list(get_mast_urls(base_url, kepid, cadence) for kepid in kepids)
    urls = {u for url in urls for u in url}
    processed_urls = get_processed_mast_urls(processed_urls_filename)
    return urls - processed_urls


def get_kepid_and_filename_from_link(link: str) -> tuple[str, str]:
    filename = link.split("/")[-1]
    kepid_str = filename.split("-")[0].replace("kplr", "")
    return kepid_str, filename


async def download_fits_file(url: str, out_dir: str) -> list[str]:
    mast_api = MastApi()
    kepid_str, filename = get_kepid_and_filename_from_link(url)

    try:
        fits_file = await mast_api.fetch(url)
    except InvalidRequestOrResponse:
        log.debug(f"{url=} not found")
    else:
        out_path = f"{out_dir}/{kepid_str}/{filename}"
        with tf.io.gfile.GFile(out_path, mode="w") as gf:
            gf.write(fits_file)
    return url


async def download_time_series(
    all_urls: set[int],
    out_dir: str,
    processed_urls_filename: str,
    num_jobs: int = 25,
    update_limit: int = 1000,
) -> None:
    processed_urls = []
    try:
        num_all_urls = len(all_urls)
        url_chunks = [all_urls[i : i + num_jobs] for i in range(0, num_all_urls, num_jobs)]

        with ProgressBar() as pbar:
            bar_task_id = pbar.add_task("Downloading FITS files", total=num_all_urls)

            for urls in url_chunks:
                num_urls = len(urls)
                tasks = [asyncio.create_task(download_fits_file(url, out_dir)) for url in urls]
                processed_urls.extend([await task for task in asyncio.as_completed(tasks)])

                # Save urls after each `update_limit` and at the end of iterations
                if len(processed_urls) % update_limit == 0 or num_urls < num_jobs:
                    data = "\n".join(processed_urls) + "\n"
                    gfile_write(processed_urls_filename, data, append=True)
                    update_counter = min(update_limit, num_urls)
                    log.info(f"{update_counter} processed urls saved to {processed_urls_filename}")
                    pbar.update(bar_task_id, advance=update_counter)
                    processed_urls.clear()
    except asyncio.CancelledError:
        cancel_tasks(tasks)
        raise


async def download_nasa_tables(urls: set[str], out_dir: str) -> None:
    try:
        nasa_api = NasaApi()
        tasks = [asyncio.create_task(nasa_api.fetch(url)) for url in urls]

        with Status("[bold green]Downloading NASA tables...") as _:
            for task in asyncio.as_completed(tasks):
                table_name, table = await task
                out_filename = f"{out_dir}/{table_name}.csv"
                gfile_write(out_filename, table)
                print(f"Table {table_name.upper()} downloaded")
    except asyncio.CancelledError:
        cancel_tasks(tasks)
        raise


async def async_main(cfg: BaseConfig) -> None:
    # await download_nasa_tables(cfg.data.fetch.nasa_urls, cfg.data.tables_dir)

    kepids = get_kepids(cfg.data.tce_filename)
    mast_urls = get_kepler_urls(
        base_url=cfg.data.fetch.mast_base_url,
        kepids=kepids,
        cadence=cfg.data.cadence,
        processed_urls_filename=cfg.data.fetch.processed_urls_filename,
    )

    log.info(f"{len(mast_urls)} URLs found")
    await download_time_series(
        all_urls=list(mast_urls),
        out_dir=cfg.data.time_series_dir,
        processed_urls_filename=cfg.data.fetch.processed_urls_filename,
        num_jobs=cfg.data.fetch.num_parallel_requests,
    )


@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: BaseConfig) -> None:
    prepare_loop(asyncio.get_event_loop())
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
