import sys
import numpy as np
from flushot.entity.artifact_entity import ClassificationMetricArtifact
from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def get_classification_score(y_true, y_pred, y_pred_proba=None) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred, average="macro")
        model_recall_score = recall_score(y_true, y_pred, average="macro")
        model_precision_score = precision_score(y_true, y_pred, average="macro")
        
        # Use probabilities for ROC AUC if provided
        if y_pred_proba is not None:
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                aucs = []
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) < 2:
                        aucs.append(np.nan)
                    else:
                        aucs.append(roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
                model_roc_auc_score = np.nanmean(aucs)
            else:
                if len(np.unique(y_true)) < 2:
                    model_roc_auc_score = np.nan
                else:
                    model_roc_auc_score = roc_auc_score(y_true, y_pred_proba)
        else:
            # fallback: use class labels (not recommended for ROC AUC)
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                model_roc_auc_score = roc_auc_score(y_true, y_pred, average="macro")
            else:
                model_roc_auc_score = roc_auc_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            roc_auc_score=model_roc_auc_score
        )
        return classification_metric
    except Exception as e:
        raise FluShotException(e, sys)