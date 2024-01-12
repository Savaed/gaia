import asyncio
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import google.cloud.dataproc_v1 as dataproc
import hydra
from google.cloud import storage
from omegaconf import OmegaConf

from gaia.config import AppConfig, DataprocServerlessConfig
from gaia.io import copy_to_gcp
from gaia.log import logger
from gaia.scripts.create_features import GCP_CONFIG_ARCHIVE_NAME


# TODO: rebuild docker image and push automatically/when config set?

load_dotenv()


async def copy_batch_files_to_gcp(
    script_path: Path, config_path: Path, staging_bucket: str
) -> tuple[str, str]:
    """Copy local Python script and hydra config files to Dataproc serverless staging bucket.

    Args:
         script_path (Path): Local script filepath
         config_path (Path): Local configuration directory path
         gcs_staging_bucket (str): Dataproc serverless GCS bucket

     Returns:
         tuple[str, set[str]]: GCS locations of the script, and set of config filepaths
    """
    script_gcs_path = f"gs://{staging_bucket}/{script_path.name}"
    await copy_to_gcp(script_path, script_gcs_path)  # Copy script

    # Copy config
    with tempfile.NamedTemporaryFile() as tmp_file:
        _, _, fmt = GCP_CONFIG_ARCHIVE_NAME.partition(".")
        shutil.make_archive(tmp_file.name, fmt, root_dir=config_path)
        config_archive_path = f"gs://{staging_bucket}/{GCP_CONFIG_ARCHIVE_NAME}"
        await copy_to_gcp(f"{tmp_file.name}.{fmt}", config_archive_path)

    # logger.info(
    #     "PySpark script copied",
    #     local_path=script_path.absolute().as_posix(),
    #     gcs_path=script_gcs_path,
    # )

    return script_gcs_path, config_archive_path


async def create_pyspark_batch(cfg: DataprocServerlessConfig) -> str:
    """Create Google Dataproc serverless batch workload.

    See: https://cloud.google.com/dataproc-serverless/docs/overview

    Args:
        cfg (DataprocServerlessConfig): Batch configuration
    """
    script_uri, config_uris = await copy_batch_files_to_gcp(
        cfg.main_script_uri, cfg.config_dir, cfg.execution_config.staging_bucket
    )
    batch = dataproc.Batch()
    batch.pyspark_batch.main_python_file_uri = script_uri
    batch.pyspark_batch.archive_uris = [config_uris]
    batch.runtime_config = cfg.runtime_config.model_dump()
    batch.environment_config.execution_config = cfg.execution_config.model_dump(exclude_unset=True)

    project_id = storage.Client().project
    region = cfg.region

    create_batch_request = dataproc.CreateBatchRequest()
    create_batch_request.parent = f"projects/{project_id}/locations/{region}"
    create_batch_request.batch = batch

    controller = dataproc.BatchControllerClient(
        client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"},
    )
    batch_metadata = controller.create_batch(create_batch_request).metadata

    batch_id: str = batch_metadata.batch_uuid
    batch_gcp_console_url = f"https://console.cloud.google.com/dataproc/batches/{region}/{batch_id}/monitoring?project={project_id}"  # noqa

    logger.bind(id=batch_id, gcp_console_url=batch_gcp_console_url).info("PySpark batch submitted")
    return batch_id


def run_local_script(cfg: AppConfig, script_path: str | Path) -> int:
    """Run local Python script.

    The user must have access to the script, script must have `main()` that return `int` value.

    Args:
        cfg (AppConfig): Configuration to use in launched script
        script_path (str | Path): Filepath to the script

    Returns:
        int: Script returned value
    """
    module_name = "spark_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.main(cfg)


async def main(cfg: AppConfig) -> int:
    cfg_copy = cfg.copy()  # Use copy of the original model to pass to run_local_script()
    cfg = AppConfig(**OmegaConf.to_object(cfg))

    if cfg.create_features.dataproc:
        await create_pyspark_batch(cfg.create_features.dataproc)
        return 0

    return run_local_script(cfg_copy, cfg.create_features.script_path.absolute())


@logger.catch(message="Unexpected error occurred")
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main_wrapper(cfg: AppConfig) -> int:
    return asyncio.run(main(cfg))


if __name__ == "__main__":
    raise SystemExit(main_wrapper())


# TODO: jak przekazać hydra config/ do dataproc batch? narazie tylko .zip przesyła
