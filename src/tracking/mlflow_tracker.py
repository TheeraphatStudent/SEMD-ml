import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import time

from core import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.experiment_name = settings.mlflow_experiment_name

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {str(e)}")
            self.experiment_id = None

        self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None) -> str:
        if run_name is None:
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags={
                'mlflow.note.content': 'this experiment doing something!'
            }
        )

        logger.info(
            f"Started MLflow run: {run_name} (ID: {self.active_run.info.run_id})")

        return self.active_run.info.run_id

    def log_params(self, params: Dict[str, Any]):
        if self.active_run is None:
            logger.warning('No active run. Call start_run() first.')
            return

        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {str(e)}")

    def log_artifact(self, artifact_path: str, artifact_path_in_run: Optional[str] = None):
        if self.active_run is None:
            logger.warning('No active run. Call start_run() first.')
            return

        try:
            if artifact_path_in_run:
                mlflow.log_artifact(artifact_path, artifact_path_in_run)
            else:
                mlflow.log_artifact(artifact_path)

            logger.info(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging artifact {artifact_path}: {str(e)}")

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        if self.active_run is None:
            logger.warning('No active run. Call start_run() first.')
            return

        X_train = dataset_info.get('X_train', [])
        X_test = dataset_info.get('X_test', [])

        balanced_dataset_size = len(X_train) + len(X_test)

        params = {
            'original_dataset_size': dataset_info.get('imbalance_info', {}).get('total_samples', 0),
            'balanced_dataset_size': balanced_dataset_size,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_features': len(dataset_info.get('feature_names', [])),
            'balancing_method': dataset_info.get('balancing_method', 'none'),
            'original_imbalance_ratio': dataset_info.get('imbalance_info', {}).get('imbalance_ratio', 0)
        }

        self.log_params(params)

        if 'imbalance_info' in dataset_info:
            original_class_counts = dataset_info['imbalance_info'].get(
                'class_counts', {})
            for cls, count in original_class_counts.items():
                mlflow.log_metric(f"original_class_count_{cls}", count)

        if len(X_train) > 0:
            from collections import Counter
            y_train = dataset_info.get('y_train', [])
            y_test = dataset_info.get('y_test', [])

            if len(y_train) > 0 and len(y_test) > 0:
                balanced_labels = list(y_train) + list(y_test)
                balanced_class_counts = Counter(balanced_labels)

                for cls, count in balanced_class_counts.items():
                    mlflow.log_metric(f"balanced_class_count_{cls}", count)

                if len(balanced_class_counts) > 1:
                    max_count = max(balanced_class_counts.values())
                    min_count = min(balanced_class_counts.values())
                    balanced_imbalance_ratio = max_count / \
                        min_count if min_count > 0 else float('inf')
                    mlflow.log_metric('balanced_imbalance_ratio',
                                      balanced_imbalance_ratio)

    def log_training_results(self, results: Dict[str, Any]):
        if self.active_run is None:
            logger.warning('No active run. Call start_run() first.')
            return

        for algorithm, result in results.items():
            if 'error' in result:
                continue

            prefix = f"{algorithm}_"

            if 'params' in result:
                for key, value in result['params'].items():
                    self.log_params({f"{prefix}param_{key}": value})

            if 'cv_score' in result:
                mlflow.log_metric(f"{prefix}cv_score", result['cv_score'])

            if 'metrics' in result:
                metrics = result['metrics']
                mlflow.log_metric(f"{prefix}accuracy",
                                  metrics.get('accuracy', 0))
                mlflow.log_metric(f"{prefix}precision",
                                  metrics.get('precision', 0))
                mlflow.log_metric(f"{prefix}recall", metrics.get('recall', 0))
                mlflow.log_metric(f"{prefix}f1", metrics.get('f1', 0))

    def register_model(
        self,
        model: Any,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
        alias: Optional[str] = None,
        X_sample: Optional[Any] = None,
        input_example: Optional[Any] = None,
        description: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if self.active_run is None:
            logger.warning('No active run. Call start_run() first.')
            return None

        try:
            signature = None
            if X_sample is not None:
                try:
                    predictions = model.predict(X_sample)
                    signature = infer_signature(X_sample, predictions)
                    logger.info('Model signature inferred for registration')
                except Exception as sig_error:
                    logger.warning(
                        f"Could not infer signature: {str(sig_error)}")

            log_kwargs = {
                'sk_model': model,
                'name': model_name,
                'registered_model_name': model_name,
            }

            if signature is not None:
                log_kwargs['signature'] = signature
            if input_example is not None:
                log_kwargs['input_example'] = input_example

            model_info = mlflow.sklearn.log_model(**log_kwargs)
            model_uri = model_info.model_uri

            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(f"model_{key}", str(value))

            try:
                model_version = self._get_latest_model_version(model_name)

                if model_version:
                    if tags:
                        for key, value in tags.items():
                            self.client.set_model_version_tag(
                                name=model_name,
                                version=model_version,
                                key=key,
                                value=str(value)
                            )

                    if description:
                        self.client.update_model_version(
                            name=model_name,
                            version=model_version,
                            description=description
                        )

                    if alias:
                        self.client.set_registered_model_alias(
                            name=model_name,
                            alias=alias,
                            version=model_version
                        )
                        logger.info(
                            f"Set alias '{alias}' -> version {model_version}")

            except Exception as registry_error:
                logger.warning(
                    f"Model logged but registry operations failed: {str(registry_error)}")

            logger.info(f"Successfully registered model: {model_name}")
            return {
                'model_uri': model_uri,
                'model_name': model_name,
                'run_id': self.active_run.info.run_id,
                'version': model_version
            }

        except Exception as e:
            try:
                logger.warning(
                    f"Model registration failed, logging model only: {str(e)}")
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name=model_name
                )

                model_uri = model_info.model_uri

                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(f"model_{key}", str(value))

                logger.info('Model logged successfully without registration')
                return {
                    'model_uri': model_uri,
                    'model_name': model_name,
                    'run_id': self.active_run.info.run_id,
                    'version': None
                }

            except Exception as fallback_error:
                logger.error(f"Failed to log model: {str(fallback_error)}")
                return None

    def _get_latest_model_version(self, model_name: str) -> Optional[str]:
        try:
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'",
                order_by=['version_number DESC'],
                max_results=1
            )
            if versions:
                return versions[0].version
        except Exception as e:
            logger.warning(f"Could not get latest model version: {str(e)}")
        return None

    def end_run(self, status: str = 'FINISHED'):
        if self.active_run is None:
            logger.warning('No active run to end.')
            return

        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")
        self.active_run = None

    def log_error(self, error_message: str, error_type: str = 'general', additional_info: Optional[Dict[str, Any]] = None):
        if self.active_run is None:
            logger.warning('No active run. Cannot log error.')
            return

        try:
            timestamp = int(time.time() * 1000)

            mlflow.log_metric(f"error_{error_type}", 1, step=timestamp)

            error_text = f"{error_type}: {error_message}"

            if additional_info:
                error_details = ', '.join(
                    [f"{k}={v}" for k, v in additional_info.items()])
                error_text += f" | {error_details}"

            mlflow.set_tag(f"error_{timestamp}", error_text[:500])

            logger.error(
                f"Logged error to MLflow: {error_type} - {error_message}")
        except Exception as e:
            logger.error(f"Failed to log error to MLflow: {str(e)}")

    def evaluate_model(self, model, X_test, y_test, model_name: Optional[str] = None):
        if self.active_run is None:
            logger.warning('No active run. Call start_run() first.')
            return None

        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
            import pandas as pd

            try:
                predictions = model.predict(X_test)
                logger.info(
                    f"Model predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
            except Exception as pred_error:
                logger.error(f"Prediction failed: {str(pred_error)}")
                logger.info(
                    f"X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'Unknown'}")
                logger.info(f"Model type: {type(model)}")
                return None

            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_proba = model.predict_proba(X_test)
                    logger.info(
                        'Successfully obtained prediction probabilities')
                except Exception as proba_error:
                    logger.warning(
                        f"Could not get prediction probabilities: {str(proba_error)}")

            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, predictions, average='weighted')

            eval_metrics = {
                'eval_accuracy': accuracy,
                'eval_precision': precision,
                'eval_recall': recall,
                'eval_f1': f1
            }

            for metric_name, metric_value in eval_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            class_report = classification_report(y_test, predictions)
            mlflow.log_text(class_report, 'classification_report.txt')

            logger.info(
                f"Model evaluation completed - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

            return {
                'metrics': eval_metrics,
                'predictions': predictions,
                'probabilities': prediction_proba,
                'classification_report': class_report
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            logger.error(
                f"X_test type: {type(X_test)}, shape: {getattr(X_test, 'shape', 'No shape attr')}")
            logger.error(
                f"y_test type: {type(y_test)}, shape: {getattr(y_test, 'shape', 'No shape attr')}")
            return None


mlflow_tracker = MLflowTracker()
