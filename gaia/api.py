""""REST API for MAST and NASA data archives."""

from typing import Optional

from gaia.constants import get_quarter_prefixes
from gaia.enums import Cadence


def create_nasa_url(
    base_url: str,
    table: str,
    data_fmt: str = "csv",
    select: Optional[list[str]] = None,
    query: Optional[str] = None,
) -> str:
    """
    Create NASA HTTP URL for scalar data.

    Parameters
    ----------
    base_url : str
        Base URL shared by each API request
    table : str
        Table name to download
    data_fmt : str, optional
        Format of downloaded tabel, by default "csv"
    select : Optional[str], optional
        Table columns to include, by default None
    query : Optional[str], optional
        SQL-like query to filter the data, by default None

    Returns
    -------
    str
        URL based on a table name, data format, selected columns and filter statement
    """
    if not base_url or not table or not data_fmt:
        raise ValueError("Parameters 'base_url', 'table', 'data_fmt' cannot be empty")

    url = f"{base_url}?table={table}&format={data_fmt}"

    if select:
        url += f"&select={','.join(select)}"

    if query:
        url += f"&where={query}"

    return url


def get_mast_urls(base_url: str, kepid: int, cadence: Cadence) -> list[str]:
    """
    Create MAST HTTP URLs for time series.

    Parameters
    ----------
    base_url : str
        Base URL shared by each API request
    kepid : int
        Id of target Kepler Object of Interest (KOI)
    cadence : Cadence
        Observation frequency

    Yields
    ------
    Generator[None, None, str]
        URL for specific KOI  that can be use to retrive time series from the MAST archive
    """
    if not base_url:
        raise ValueError("'base_url' cannot be empty")

    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 to 999 999 999 inclusive, but got {kepid=}")

    fmt_kepid = f"{kepid:09}"
    url = (
        f"{base_url}?uri=mast:Kepler/url/missions/kepler/lightcurves/"
        f"{fmt_kepid[:4]}/{fmt_kepid}//kplr{fmt_kepid}"
    )
    return [f"{url}-{prefix}_{cadence.value}.fits" for prefix in get_quarter_prefixes(cadence)]


# class BaseApi(Protocol):
#     async def download(self, url: str) -> AnyStr:
#         ...


# class MastApi:
#     async def download(self, url: str) -> AnyStr:
#         raise NotImplementedError


# class NasaApi:
#     async def download(self, url: str) -> AnyStr:
#         raise NotImplementedError

#     def _split_url(self, url: str) -> dict[str, str]:
#         pass

#     def _validate_response(self, response: AnyStr) -> None:
#         pass
