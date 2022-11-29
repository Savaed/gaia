"""I/O functions to use in local or cloud environments (GCP, AWS) or HDFS."""

from enum import Enum
from typing import AnyStr

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError, PermissionDeniedError


class FileMode(Enum):
    WRITE = "wb"
    WRITE_BINARY = "w"
    READ = "r"
    READ_BINARY = "rb"
    APPEND = "a"


def read(src: str, mode: FileMode = FileMode.READ_BINARY) -> AnyStr:
    """Read file content from local environment cloud (GCP, AWS) or HDFS.

    Args:
        src (str): Path to the source file
        mode (FileMode, optional): Read mode. Defaults to `FileMode.READ_BINARY`.

    Raises:
        FileNotFoundError: The requested file was not found
        PermissionError: No permissions for file or cloud environment

    Returns:
        AnyStr: File content as bytes or string
    """
    try:
        with tf.io.gfile.GFile(src, mode.value) as gf:
            return gf.read()  # type: ignore
    except NotFoundError:
        raise FileNotFoundError(src)
    except PermissionDeniedError as ex:
        raise PermissionError(ex)


def write(dest: str, data: AnyStr, mode: FileMode = FileMode.WRITE_BINARY) -> None:
    """Write data to a file in local environment, cloud (GCP, AWS) or HDFS.

    Args:
        dest (str): Path to the destination file. It must exist
        data (AnyStr): Data to write, bytes or string
        mode (FileMode, optional): Write mode. Defaults to `FileMode.WRITE_BINARY`.

    Raises:
        FileNotFoundError: The requested file was not found
        PermissionError: No permissions for file or cloud environment
    """
    try:
        with tf.io.gfile.GFile(dest, mode.value) as gf:
            gf.write(data)
    except NotFoundError:
        raise FileNotFoundError(dest)
    except PermissionDeniedError as ex:
        raise PermissionError(ex)
