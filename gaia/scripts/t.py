import asyncio
from gaia.io import copy_to_gcp
import dotenv


dotenv.load_dotenv()


async def main():
    await copy_to_gcp(
        "/home/krzysiek/projects/gaia/data/kepler/interim/", "gs://gaia-data-kepler/interim"
    )


if __name__ == "__main__":
    asyncio.run(main())
