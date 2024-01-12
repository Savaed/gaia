import json
from pathlib import Path

from google.cloud import service_usage_v1 as service_usage

from gaia.config import GCPRegion
from gaia.log import logger


def main() -> int:
    service_name = "storage.googleapis.com"
    project_number = 967263789036

    service = f"projects/{project_number}/services/{service_name}"

    client = service_usage.ServiceUsageClient()

    # Get service
    request = service_usage.GetServiceRequest()
    request.name = service
    response = client.get_service(request)
    print(response)

    # Enable service
    request = service_usage.EnableServiceRequest()
    request.name = service
    response = client.enable_service(request)
    print(response)

    # Disable service
    request = service_usage.DisableServiceRequest()
    request.name = service
    request.disable_dependent_services = True
    response = client.disable_service(request)
    print(response)
    return 0


def add_gcp_to_docker(region: GCPRegion) -> None:
    """Add Google Cloud Artifact Registry to Docker configuration.

    Args:
        region (GCPRegion): Region which host Artifact Registry
    """
    docker_config_filepath = Path.home() / ".docker" / "config.json"
    docker_config_filepath.touch(0o600)  # -rw- --- ---

    try:
        docker_config: dict = json.loads(docker_config_filepath.read_text())
        logger.warning(
            "Docker config file contains configuration", path=docker_config_filepath.as_posix()
        )
    except json.JSONDecodeError:
        docker_config = {}

    docker_config["credHelpers"] = docker_config.get("credHelpers", {}) | {
        f"{region}-docker.pkg.dev": "gcloud"
    }

    docker_config_filepath.write_text(json.dumps(docker_config, indent=2) + "\n")


if __name__ == "__main__":
    # raise SystemExit(main())
    add_gcp_to_docker("europe-central2")
