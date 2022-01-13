import tarfile
import io

from binascii import b2a_hex
from os import urandom
from pathlib import Path


def random_id(size: int = 2):
    """
        Generates random id of specified size.
    """
    return b2a_hex(urandom(size)).decode()


def str_to_tarfile(data: str, tar_info_name: str) -> Path:
    random = b2a_hex(urandom(2)).decode()
    dest_path = Path('/tmp', random + ".tar")

    info = tarfile.TarInfo(name=tar_info_name)
    info.size = len(data)

    with tarfile.TarFile(str(dest_path), 'w') as tar:
        tar.addfile(info, io.BytesIO(data.encode('utf8')))

    return dest_path
