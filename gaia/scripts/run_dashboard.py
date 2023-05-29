from textwrap import dedent

import hydra
from dash import Dash
from omegaconf import OmegaConf

from gaia.config import Config
from gaia.data.mappers import map_kepler_stallar_parameters, map_kepler_tce, map_kepler_time_series
from gaia.data.models import KeplerStellarParameters, KeplerTCE, KeplerTimeSeries
from gaia.data.preprocessing import compute_euclidean_distance, normalize_median
from gaia.data.stores import (
    StellarParametersStore,
    StellarParametersStoreParamsSchema,
    TceStore,
    TceStoreParamsSchema,
    TimeSeriesStore,
)
from gaia.io import ParquetReader, ParquetTableReader
from gaia.ui.cli import print_header
from gaia.ui.components import TimeSeriesAIO
from gaia.ui.dashboard import STORES, create_dashboard
from gaia.utils import compose


SCRIPT_DESCRIPTION = """
    Launch a website with scalar data visualization (TCE, stellar parameters and statistics) as
    well as time series.
"""


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: Config) -> int:
    print_header(dedent(SCRIPT_DESCRIPTION))

    cfg_dict = OmegaConf.to_object(cfg)
    cfg = Config(**cfg_dict)

    app = Dash(
        name=__name__,
        assets_folder=cfg.ui.assets_dir.absolute().as_posix(),
        external_stylesheets=cfg.ui.external_stylesheets,
    )

    # TODO: Organize this in hydra config. For now it's hardcoded.
    compute_centroid_shift = compose(compute_euclidean_distance, normalize_median)
    TimeSeriesAIO.add_preprocessing("mom_centr1,mom_centr2", compute_centroid_shift)
    TimeSeriesAIO.add_preprocessing("pdcsap_flux", normalize_median)

    STORES[TimeSeriesStore] = TimeSeriesStore[KeplerTimeSeries](
        map_kepler_time_series,
        ParquetReader("/home/krzysiek/projects/gaia/data/interim/time_series"),
    )

    STORES[StellarParametersStore] = StellarParametersStore[KeplerStellarParameters](
        map_kepler_stallar_parameters,
        ParquetTableReader(
            "/home/krzysiek/projects/gaia/data/interim/tables/q1_q17_dr25_stellar2.parquet",
        ),
        StellarParametersStoreParamsSchema(id="kepid"),
    )

    STORES[TceStore] = TceStore[KeplerTCE](
        map_kepler_tce,
        ParquetTableReader(
            "/home/krzysiek/projects/gaia/data/interim/tables/q1_q17_dr25_tce_merged.parquet",
        ),
        TceStoreParamsSchema(
            target_id="kepid",
            tce_id="tce_plnt_num",
            name="kepler_name",
            label="label",
            duration="tce_duration",
            epoch="tce_time0bk",
            period="tce_period",
        ),
    )

    app.layout = create_dashboard(cfg.ui.available_graphs)
    app.run(**cfg.ui.server_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
