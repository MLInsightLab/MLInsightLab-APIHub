from global_variables import DATA_DIRECTORY
from subprocess import check_call
import base64
import os

# Upload data to data store


def upload_data_to_fs(
        filename: str,
        file_bytes: str,
        overwrite: bool = False
):
    '''
    Upload data to the data store

    Parameters:
    -----------
    filename : str
        The name of the file, either with or without /data prepended
    file_bytes : str
        The bytes of the file, encoded base64 and then to utf-8, if a binary file
    overwrite : bool (default False)
        Whether to overwrite the file if it already exists

    Returns
    -------
    filename : str
        The final filename of the file, on disk
    '''

    # Ensure that the data directory leads
    if not filename.startswith(DATA_DIRECTORY):
        filename = os.path.join(
            DATA_DIRECTORY,
            filename.lstrip('/').strip()
        )

    # If the file exists and overwrite False, then raise an Exception
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(
            'Data file already exists and overwrite was not set to True')

    # Create any intermediate directories if needed
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, mode=771)

    # Determine the content of the file
    file_content = base64.b64decode(
        file_bytes.encode('utf-8')
    )

    # Opener as wrapper function to write file
    def opener(path, flags):
        return os.open(path, flags, 0o776)

    # Write the file
    with open(filename, 'wb', opener=opener) as f:
        f.write(file_content)

    # change the group of the file so that all mlil users in Jupyter can use the file
    check_call(
        ['chgrp', 'mlil', filename]
    )

    return filename

# Download data from data store


def download_data_from_fs(
        filename: str
):
    '''
    Download a file from the file system

    Parameters
    ----------
    filename : str
        The name of the file

    Returns
    -------
    content : str
        The content of the file, as a string
    '''

    # Get the correct name of the file
    if not filename.startswith(DATA_DIRECTORY):
        filename = os.path.join(
            DATA_DIRECTORY,
            filename.lstrip('/').strip()
        )

    # Check that file exists
    if not os.path.exists(filename):
        raise FileNotFoundError('File does not exist')

    # Read the file and return the encoded contents
    with open(filename, 'rb') as f:
        content = f.read()
    content = base64.b64encode(content).decode('utf-8')

    return content

# List data in the data store


def list_fs_directory(dirname: str = None) -> list[str]:
    '''
    List the contents of a directory in the file store

    Parameters
    ----------
    dirname : str or None (default None)
        The directory name to list

    Returns
    -------
    files : list[str]
        The files in that directory
    '''

    if dirname is None:
        dirname = DATA_DIRECTORY

    if not dirname.startswith(DATA_DIRECTORY):
        dirname = os.path.join(
            DATA_DIRECTORY,
            dirname.lstrip('/').strip()
        )

    if not os.path.isdir(dirname):
        raise TypeError('No directory found')

    return os.listdir(dirname)
