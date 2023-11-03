import hydra
from dash import Dash
from omegaconf import OmegaConf

from gaia.config import AppConfig
from gaia.data.stores import StellarParametersStore, TceStore, TimeSeriesStore
from gaia.log import logger
from gaia.ui.cli import print_header
from gaia.ui.components import PreprocessingFunction, TimeSeriesAIO
from gaia.ui.dashboard import STORES, create_dashboard


@logger.catch(message="Unexpected error occurred")
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: AppConfig) -> int:
    cfg = AppConfig(**OmegaConf.to_object(cfg))
    print_header(cfg.ui.script_description)

    for graph_id, preprocessing_func in cfg.ui.graphs_preprocessors.items():
        preprocessor: PreprocessingFunction = hydra.utils.instantiate(preprocessing_func)
        TimeSeriesAIO.add_preprocessing(graph_id, preprocessor)

    STORES[TimeSeriesStore] = hydra.utils.instantiate(cfg.data_providers.time_series)
    STORES[TceStore] = hydra.utils.instantiate(cfg.data_providers.tce)
    STORES[StellarParametersStore] = hydra.utils.instantiate(cfg.data_providers.stellar_params)

    app = Dash(
        name=__name__,
        assets_folder=cfg.ui.assets_dir.absolute().as_posix(),  # Must be absolute path
        external_stylesheets=cfg.ui.external_stylesheets,
    )
    app.layout = create_dashboard(
        available_time_series_graphs=cfg.ui.available_graphs,
        stellar_parameters_units=cfg.ui.stellar_parameters_units,
        planetary_parameters_units=cfg.ui.planetary_parameters_units,
    )
    app.run(**cfg.ui.server_params.model_dump())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
