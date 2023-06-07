import asyncio

import hydra
from omegaconf import OmegaConf

from gaia.asynchronous import prepare_loop
from gaia.config import AppConfig
from gaia.log import logger
from gaia.ui.cli import print_header


async def main(cfg: AppConfig) -> int:
    prepare_loop(asyncio.get_running_loop())
    cfg = AppConfig(**OmegaConf.to_object(cfg))
    print_header(None)

    tce_converter = hydra.utils.instantiate(cfg.preprocessing.conversion.tce.converter)
    staller_converter = hydra.utils.instantiate(
        cfg.preprocessing.conversion.stellar_parameters.converter,
    )
    fits_converter = hydra.utils.instantiate(cfg.preprocessing.conversion.time_series.converter)

    tce_converter.convert(**cfg.preprocessing.conversion.tce.conversion_parameters)
    staller_converter.convert(
        **cfg.preprocessing.conversion.stellar_parameters.conversion_parameters,
    )

    try:
        await fits_converter.convert(
            **cfg.preprocessing.conversion.time_series.conversion_parameters,
        )
    except KeyboardInterrupt:
        logger.info("Script shutdown")

    return 0


@logger.catch(message="Unexpected error occurred")
@hydra.main("../../configs", "config", version_base="1.3")
def main_wrapper(cfg: AppConfig) -> int:
    raise SystemExit(asyncio.run(main(cfg)))


if __name__ == "__main__":
    main_wrapper()
