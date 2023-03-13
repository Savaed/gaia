import gzip
import hashlib
import pickle
from typing import Any

import fakeredis
import structlog


logger = structlog.stdlib.get_logger()


class DataStoreError(Exception):
    """Raised when cannot serialize/deserialize or save/load value from the store."""


class RedisStore:
    _store = fakeredis.FakeStrictRedis()

    @classmethod
    def save(cls, data: Any) -> str:
        """Pickle, compress and save the data in an in-memory datastore.

        Arguments:
            data (any): data to serialize

        Insteps:
            DataStoreError: Unable to serialized data

        Returns:
            str: SHA256 hash under which the data is saved
        """
        try:
            pickled_data = pickle.dumps(data)
        except (pickle.PicklingError, RecursionError) as ex:
            raise DataStoreError(f"Cannot serialize data. {ex}")

        serialized_data = gzip.compress(pickled_data)
        hash_key = cls._hash(serialized_data)

        if cls._store.exists(hash_key):
            logger.warning()
            return hash_key

        cls._store.set(hash_key, serialized_data)
        logger.info("Data saved", key=hash_key)
        return hash_key

    @classmethod
    def load(cls, key: str) -> Any:
        """Deserialize data stored under the specified key.

        Arguments:
            key(str): The key. This is the SHA256 hash of the serialized data

        Insteps:
            KeyError: No data found for the specified key

        Returns:
            Any: Deserialized data
        """
        serialized_data = cls._store.get(key)
        if not serialized_data:
            raise KeyError(key)

        data = pickle.loads(gzip.decompress(serialized_data))
        logger.info("Data loaded", key=key)
        return data

    @staticmethod
    def _hash(serialized_data: bytes) -> str:
        return hashlib.sha256(serialized_data).hexdigest()
