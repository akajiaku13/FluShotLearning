import os, sys, mlflow

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging

from flushot.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from flushot.entity.config_entity import ModelTrainerConfig

from flushot.utils.main_utils.utils import (
    save_object, load_object, load_numpy_array_data, evaluate_models
)
from flushot.utils.ml_utils.metric.classification_metric import get_classification_score
from flushot.utils.ml_utils.model.estimator import NetworkModel

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import r2_score, roc_auc_score

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
       try:
           self.model_trainer_config = model_trainer_config
           self.data_transformation_artifact = data_transformation_artifact
       except Exception as e:
           raise FluShotException(e, sys)
       
    def track_mlflow(self, best_model, classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score
            roc_auc_score = classificationmetric.roc_auc_score

            mlflow.log_metric('F1_Score', f1_score)
            mlflow.log_metric('Precision', precision_score)
            mlflow.log_metric('Recall Score', recall_score)
            mlflow.log_metric('ROC AUC Score', roc_auc_score)
            mlflow.sklearn.log_model(best_model, 'model')
       
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            'Random Forest': MultiOutputClassifier(RandomForestClassifier(verbose=1)),
            'KNN': MultiOutputClassifier(KNeighborsClassifier()),
            'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier()),
            'Gradient Boosting': MultiOutputClassifier(GradientBoostingClassifier(verbose=1)),
            'Logistic Regression': MultiOutputClassifier(LogisticRegression(verbose=1, max_iter=500)),
            'AdaBoost': MultiOutputClassifier(AdaBoostClassifier())
        }

        params = {
            'Decision Tree': {
                'estimator__criterion':['gini', 'entropy', 'log_loss']
            },
            'Random Forest':{
                'estimator__n_estimators':[8,16,32,64,128,256]
            },
            'Gradient Boosting':{
                'estimator__learning_rate': [.1,.01,.05,.001],
                'estimator__subsample': [.6,.7,.75,.8,.85,.9],
                'estimator__n_estimators':[8,16,32,64,128,256]
            },
            'Logistic Regression': {},
            'AdaBoost':{
                'estimator__learning_rate': [.1,.01,.05,.001],
                'estimator__n_estimators':[8,16,32,64,128,256]
            },
            'KNN': {
                'estimator__n_neighbors': [3, 5, 7, 9, 11],
                'estimator__weights': ['uniform', 'distance'],
                'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             model=models, param=params)
        
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]
        logging.info(f'Best model for data: {best_model}')

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        # Track the MLFlow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # Track the MLFlow
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        save_object('final_model/model.pkl', best_model)

        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric)
        logging.info(f'Model Trainer Artifacts: {model_trainer_artifact}')

        return model_trainer_artifact
       
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-2],
                train_arr[:, -2:],
                test_arr[:, :-2],
                test_arr[:, -2:]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise FluShotException(e, sys)