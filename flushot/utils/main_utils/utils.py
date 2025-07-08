import yaml
import os, sys
import numpy as np
import dill
import pickle

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, roc_auc_score

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file content.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FluShotException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.
    
    :param file_path: Path to the YAML file.
    :param data: Dictionary to write to the YAML file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise FluShotException(e, sys)