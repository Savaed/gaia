import hydra
from dash import Dash
from omegaconf import OmegaConf

from gaia.config import AppConfig
from gaia.data.stores import StellarParametersStore, TceStore, TimeSeriesStore
from gaia.ui.cli import print_header
from gaia.ui.components import PreprocessingFunction, TimeSeriesAIO
from gaia.ui.dashboard import STORES, create_dashboard


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: AppConfig) -> int:
    cfg = AppConfig(**OmegaConf.to_object(cfg))
    print_header(cfg.ui.script_description)

    for graph_id, preprocessing_func in cfg.ui.graphs_preprocessing.items():
        preprocessor: PreprocessingFunction = hydra.utils.instantiate(preprocessing_func)
        TimeSeriesAIO.add_preprocessing(graph_id, preprocessor)

    STORES[TimeSeriesStore] = hydra.utils.instantiate(cfg.data.time_series_store)
    STORES[TceStore] = hydra.utils.instantiate(cfg.data.tce_store)
    STORES[StellarParametersStore] = hydra.utils.instantiate(cfg.data.stellar_store)

    app = Dash(
        name=__name__,
        assets_folder=cfg.ui.assets_dir.absolute().as_posix(),
        external_stylesheets=cfg.ui.external_stylesheets,
    )
    app.layout = create_dashboard(cfg.ui.available_graphs)
    app.run(**cfg.ui.server_params.dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
