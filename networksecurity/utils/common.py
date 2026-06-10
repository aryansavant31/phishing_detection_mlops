from box.exceptions import BoxValueError
from box import ConfigBox
import json
import joblib
from pathlib import Path
from typing import Any
import os
import yaml
from networksecurity.logging.logger import logger
from networksecurity.exceptions.custom_exception import NetworkSecurityException
import sys
import numpy as np
import pickle

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Parameters
    -----------
    path_to_yaml: Path
        path to the .yaml file
    """
    try:
        with open(path_to_yaml) as yaml_file:
            params = ConfigBox(yaml.safe_load(yaml_file))
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return params
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e
    
def write_yaml(file_path:str, content:object, replace:bool=False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logger.info(f"yaml file saved sucessfully at {file_path}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def create_directories(paths_to_dir: list, verbose=True):
    for path in paths_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Creating directory at {path}")

def save_numpy_array(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Numpy file saved sucessfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_numpy_array(file_path: str) -> np.ndarray:
    try:
        # check if file exists
        if os.path.exists(file_path):
            data_arr = np.load(file_path)
            return data_arr
        else:
            raise ValueError(f"The filepath {file_path} is not found")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object file saved sucessfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e