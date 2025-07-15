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
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Saves a NumPy array to a file.
    
    :param file_path: Path to the file where the array will be saved.
    :param array: NumPy array to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)   
    except Exception as e:
        raise FluShotException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object to a file using dill.
    
    :param file_path: Path to the file where the object will be saved.
    :param obj: Object to save.
    """
    try:
        logging.info('Entered the save_object method')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise FluShotException(e, sys) from e
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception (f'The file: {file_path} does not exist')
        with open(file_path, 'rb') as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise FluShotException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FluShotException(e, sys)

def safe_roc_auc_score(y_true, y_pred_proba):
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred_proba)
    else:
        aucs = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) < 2:
                aucs.append(np.nan)
            else:
                aucs.append(roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
        # Return the mean of valid AUCs (ignoring NaNs)
        return np.nanmean(aucs)

def evaluate_models(X_train, y_train, X_test, y_test, model, param):
    try:
        report_r2 = {}
        report_auc = {}
        for i in range(len(list(model))):
            models = list(model.values())[i]
            para = param[list(model.keys())[i]]

            gs = GridSearchCV(models, para, cv=3)
            gs.fit(X_train, y_train)

            models.set_params(**gs.best_params_)
            models.fit(X_train, y_train)

            # Use probability predictions for ROC AUC
            if hasattr(models, "predict_proba"):
                # For MultiOutputClassifier, returns list of arrays
                y_train_pred_proba = np.column_stack([
                    prob[:, 1] for prob in models.predict_proba(X_train)
                ])
                y_test_pred_proba = np.column_stack([
                    prob[:, 1] for prob in models.predict_proba(X_test)
                ])
            else:
                # fallback for models without predict_proba
                y_train_pred_proba = models.predict(X_train)
                y_test_pred_proba = models.predict(X_test)

            y_train_pred = models.predict(X_train)
            y_test_pred = models.predict(X_test)

            train_model_score_r2 = r2_score(y_train, y_train_pred)
            test_model_score_r2 = r2_score(y_test, y_test_pred)

            report_r2[list(model.keys())[i]] = test_model_score_r2

            train_model_score_auc = safe_roc_auc_score(y_train, y_train_pred_proba)
            test_model_score_auc = safe_roc_auc_score(y_test, y_test_pred_proba)

            report_auc[list(model.keys())[i]] = test_model_score_auc

        return report_auc
    except Exception as e:
        raise FluShotException(e, sys)