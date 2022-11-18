"""
Helper functions used for dataset preparation and handling.
"""
import hashlib
import tarfile
from zipfile import ZipFile

import requests
from tqdm import tqdm


class HashValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedFileTypeError(Exception):
    def __init__(self, message):
        super().__init__(message)


def extract_tar(file_path, out_path):
    """
    Extracts the contents of a tar file.

    :param file_path: Input tar file.
    :param out_path: Output directory
    """
    with tarfile.open(file_path, 'r') as f:
        f.extractall(out_path)


def extract_zip(file_path, out_path):
    """
    Extracts the contents of a zip file.

    :param file_path: Input zip file.
    :param out_path: Output directory
    """
    with ZipFile(file_path, 'r') as f:
        f.extractall(out_path)


def extract_archive(file_path, out_path):
    """
    Extracts the contents of a archive file. Supports zip, tar and tar.gz files.

    :param file_path: Input archive file.
    :param out_path: Output directory.
    :raises UnsupportedFileTypeError: If the archive type is not supported.
    """
    mapping = {
        '.zip': extract_zip,
        '.tar': extract_tar,
        '.tar.gz': extract_tar,
        '.tgz': extract_tar,
    }

    for ext, fun in mapping.items():
        if file_path.endswith(ext):
            fun(file_path, out_path)
            return

    raise UnsupportedFileTypeError(f'Unsupported archive type. Supported file types are [{" ".join(mapping.keys())}].')


def check_md5(file_path, md5_hash):
    """
    Computes the md5 hash of the given file and compares it with the given hash value.
    :param file_path: Input file path.
    :param md5_hash: Expected input file hash.
    :raises HashValidationError: When the hashes mismatch.
    """
    check_hash(file_path, md5_hash, 'md5')


def check_hash(file_path, hash_value, algo):
    """
    Computes a file hash and compares it to the given expected hash value.

    :param file_path: Input file path.
    :param hash_value: Expected hash value.
    :param algo: Hashing algorithm. See hashlib for supported algorithms.
    :raises HashValidationError: When the hashes mismatch.
    :raises ValueError: When the given hashing algorthm is not supported.
    """
    hash_fun = getattr(hashlib, algo)
    if hash_fun is None:
        raise ValueError(f'Invalid hashing algorithm: {algo}.')

    with open(file_path, 'rb') as f:
        actual_md5 = hash_fun(f.read()).hexdigest()

        if actual_md5.lower() != hash_value.lower():
            raise HashValidationError(f'Hash mismatch for file {file_path}. '
                                      f'Expected hash: {hash_value.lower()}, actual hash: {actual_md5.lower()}')


def download_url(url, out_file, description, chunk_size=4096):
    """
    Downloads the file from the given url and puts it at the output path location.

    :param url: File url to download.
    :param out_file: Path to the output file.
    :param description: Progress indicator description string.
    :param chunk_size: Download chunk size.
    :raises RuntimeError: If the server responded with a not ok code.
    """
    r = requests.get(url, stream=True)
    if not r.ok:
        err_msg = f'Could not download file from {url}.'
        raise RuntimeError(err_msg)

    content_len = int(r.headers.get('content-length', 0))

    with open(out_file, 'wb') as f:
        progress = tqdm(total=content_len, desc=description, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            progress.update(chunk_size)
