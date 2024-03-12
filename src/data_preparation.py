"""
This script contains functionalities to download and prepare the project datasets
"""
import os
import zipfile
import gdown
import zipfile
import shutil

from pathlib import Path
from typing import Union, Optional

HOME = os.path.dirname(os.path.realpath(__file__))
current = HOME 
while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
DATA_FOLDER = os.path.join(PARENT_DIR, 'data')

# utility functions to manipulate files and modify the file system
def abs_path(path: Union[str, Path]) -> Path:
    return Path(path) if os.path.isabs(path) else Path(os.path.join(HOME, path))

def squeeze_directory(directory_path: Union[str, Path]) -> None:
    # Given a directory with only one subdirectory, this function moves all the content of
    # subdirectory to the parent directory

    # first convert to the absolute path
    path = abs_path(directory_path)

    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    if len(files) == 1 and os.path.isdir(os.path.join(path, files[0])):
        subdir_path = os.path.join(path, files[0])
        # copy all the files in the subdirectory to the parent one
        for file_name in os.listdir(subdir_path):
            shutil.move(src=os.path.join(subdir_path, file_name), dst=path)
        # done forget to delete the subdirectory
        os.rmdir(subdir_path)

def unzip_data_file(data_zip_path: Union[Path, str],
                    unzip_directory: Optional[Union[Path, str]] = None,
                    remove_inner_zip_files: bool = True) -> Union[Path, str]:
    data_zip_path = abs_path(data_zip_path)

    assert os.path.exists(data_zip_path), "MAKE SURE THE DATA'S PATH IS SET CORRECTLY!!"

    if unzip_directory is None:
        unzip_directory = Path(data_zip_path).parent

    unzipped_dir = os.path.join(unzip_directory, os.path.basename(os.path.splitext(data_zip_path)[0]))
    os.makedirs(unzipped_dir, exist_ok=True)

    # let's first unzip the file
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        # extract the data to the unzipped_dir
        zip_ref.extractall(unzipped_dir)

    # unzip any files inside the subdirectory
    for file_name in os.listdir(unzipped_dir):
        file_path = os.path.join(unzipped_dir, file_name)
        
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # extract the data to current directory
                zip_ref.extractall(unzipped_dir)
            
            zip_ref.close()

    # remove any zip files left
    for file_name in os.listdir(unzipped_dir):
        basename, ext = os.path.splitext(file_name)
    
        if ext == '.zip':
            os.remove(os.path.join(unzipped_dir, file_name))                                    


    # squeeze all the directories
    for file_name in os.listdir(unzipped_dir):
        squeeze_directory(os.path.join(unzipped_dir, file_name))

    return unzipped_dir


def _download_data():
    # download the data
    url = 'https://drive.google.com/file/d/1MrgVczAKBKCPYnTXOWyieeOSDwM-ijUs/view?usp=drive_link'
    # create the data folder is needed
    os.makedirs(DATA_FOLDER, exist_ok=True)
    output = os.path.join(DATA_FOLDER, 'archive.zip')
    gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
    unzip_data_file(output, unzip_directory=os.path.join(DATA_FOLDER))
    os.remove(output)
    squeeze_directory(os.path.join(DATA_FOLDER))


def prepare_data():
    ready = False
    # keep trying to download the data until the process terminates successfully
    while not ready:
        try: 
            _download_data()
        except:
            continue
        ready = True
