import tomllib
from pathlib import Path

import hydra
from rich.console import Console


GAIA_HEADER = """
 ██████╗  █████╗ ██╗ █████╗
██╔════╝ ██╔══██╗██║██╔══██╗
██║  ███╗███████║██║███████║
██║   ██║██╔══██║██║██╔══██║
╚██████╔╝██║  ██║██║██║  ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
"""


def print_header(description: str | None = None, hydra_info: bool = True) -> None:
    """Display project main information (project description and version from 'pyproject.toml').

    Args:
        description (str | None, optional): Detailed description. Defaults to None.
        hydra_info (bool, optional): Whether to print a config path. Hydra config must be set.
            Defaults to True.
    """
    project_meta_filename = Path(__file__).parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(project_meta_filename.read_text())
    project_section = pyproject["tool"]["poetry"]
    project_version = project_section["version"]
    project_description = project_section["description"]

    with Console(width=120) as console:
        console.print(f"[green]{GAIA_HEADER}")
        console.print(f"v. {project_version}\n")
        console.print(f"🌍 [bold]{project_description}[/bold]\n")

        if description:
            console.print(f"{description.strip()}\n")

        if hydra_info:
            try:
                hydra_config_path = Path(
                    hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"],
                )
            except ValueError:
                console.print("❌ Hydra configuration is not set")
            else:
                console.print(
                    f"📜 Script configuration can be found at: {hydra_config_path / '.hydra'/ 'config.yaml'}",  # 'config.yaml' by convention, but may change # noqa
                )

        console.rule()
