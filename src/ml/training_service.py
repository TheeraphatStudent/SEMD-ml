import os
import json
from typing import Dict, Any, List
from datetime import datetime
import logging

from core import settings
from data import dataset_pipeline
from ml.ml_pipeline import ml_pipeline
from tracking import mlflow_tracker
from infra import db_client
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingService:

    def __init__(self):
        self.reports_path = settings.reports_path
        os.makedirs(self.reports_path, exist_ok=True)

    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def execute_training(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting training job: {job_data}")

        start_time = datetime.now()

        try:
            service_conf_id = job_data.get('service_conf_id')
            dataset_files = job_data.get('dataset_files', [])
            algorithms = job_data.get(
                'algorithms', settings.available_algorithms)
            run_name = job_data.get(
                'run_name', f"training_{start_time.strftime('%Y%m%d_%H%M%S')}")
            balance_method = job_data.get('balance_method')

            if not dataset_files:
                raise ValueError('No dataset files provided')

            logger.info('\n- Step 1: Preparing dataset...')
            dataset_result = dataset_pipeline.prepare_dataset(
                dataset_files=dataset_files,
                apply_balancing=True,
                manual_balance_method=balance_method
            )

            X_train = dataset_result['X_train']
            X_test = dataset_result['X_test']
            y_train = dataset_result['y_train']
            y_test = dataset_result['y_test']

            logger.info('\n- Step 2: Starting MLflow run...')
            run_id = mlflow_tracker.start_run(run_name=run_name)

            mlflow_tracker.log_dataset_info(dataset_result)

            logger.info('\n- Step 3: Preprocessing data...')
            X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = ml_pipeline.preprocess_data(
                X_train, X_test, y_train, y_test
            )

            if settings.enable_feature_selection:
                logger.info(
                    '\n- Step 4: Feature selection pipeline...')
                X_train_df = pd.DataFrame(
                    X_train_scaled, columns=dataset_result['feature_names'])
                X_train_selected, selected_features = ml_pipeline.feature_selection(
                    X_train_df, y_train_encoded, dataset_result['feature_names']
                )

                X_test_df = pd.DataFrame(
                    X_test_scaled, columns=dataset_result['feature_names'])
                X_test_selected = X_test_df[selected_features].values

                logger.info(
                    f"Selected {len(selected_features)} features from {len(dataset_result['feature_names'])}")
            else:
                logger.info(
                    '\n- Step 4: Skipping feature selection (disabled in config)')
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
                selected_features = dataset_result['feature_names']

            logger.info('\n- Step 5: Training and comparing models...')
            logger.info(
                f"Final training feature count: {X_train_selected.shape[1]}")

            training_results = ml_pipeline.train_and_compare_models(
                X_train_selected, X_test_selected, y_train_encoded, y_test_encoded,
                algorithms=algorithms
            )

            if 'error' in training_results:
                raise ValueError(
                    f"Training failed: {training_results['error']}")

            mlflow_tracker.log_training_results(training_results)

            logger.info('\n- Step 6: Saving artifacts...')
            artifacts = ml_pipeline.save_artifacts(run_id)

            for artifact_path in artifacts.values():
                mlflow_tracker.log_artifact(artifact_path)

            logger.info('\n- Step 7: Registering best model...')
            best_algorithm = ml_pipeline.best_algorithm
            best_metrics = training_results[best_algorithm]['metrics']

            model_name = f"malicious_url_detector_{best_algorithm}"
            registration_result = mlflow_tracker.register_model(
                ml_pipeline.best_model,
                model_name,
                tags={
                    'algorithm': best_algorithm,
                    'f1_score': str(best_metrics['f1']),
                    'accuracy': str(best_metrics['accuracy'])
                },
                alias='champion',
                X_sample=X_test_selected[:5] if len(X_test_selected) > 0 else None
            )
            model_uri = registration_result.get('model_uri') if registration_result else None

            evaluation_result = mlflow_tracker.evaluate_model(
                model=ml_pipeline.best_model,
                X_test=X_test_selected,
                y_test=y_test_encoded,
                model_name=model_name
            )

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            logger.info('\n- Step 8: Generating report...')
            report = self._generate_report(
                run_id=run_id,
                training_results=training_results,
                dataset_result=dataset_result,
                training_time=training_time,
                best_algorithm=best_algorithm,
                artifacts=artifacts,
                selected_features=selected_features if 'selected_features' in locals() else None
            )

            report_path = os.path.join(
                self.reports_path, f"{best_algorithm}_training_report_{run_id}.json")
            report = self._convert_numpy_types(report)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            mlflow_tracker.log_artifact(report_path)

            logger.info('\n- Step 9: Updating database...')
            if service_conf_id:
                self._update_model_registry(
                    service_conf_id=service_conf_id,
                    run_id=run_id,
                    best_algorithm=best_algorithm,
                    best_metrics=best_metrics,
                    artifacts=artifacts,
                    model_uri=model_uri
                )

            mlflow_tracker.end_run(status='FINISHED')

            logger.info(
                f"Training completed successfully in {training_time:.2f} seconds")

            return {
                'status': 'success',
                'run_id': run_id,
                'best_algorithm': best_algorithm,
                'metrics': best_metrics,
                'training_time': training_time,
                'report': report,
                'artifacts': artifacts
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)

            mlflow_tracker.log_error(
                error_message=str(e),
                error_type='training_failure',
                additional_info={
                    'service_conf_id': job_data.get('service_conf_id'),
                    'dataset_files': job_data.get('dataset_files', []),
                    'algorithms': job_data.get('algorithms', []),
                    'run_name': job_data.get('run_name', f"training_{start_time.strftime('%Y%m%d_%H%M%S')}")
                }
            )

            if mlflow_tracker.active_run:
                mlflow_tracker.end_run(status='FAILED')

            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_report(
        self,
        run_id: str,
        training_results: Dict[str, Any],
        dataset_result: Dict[str, Any],
        training_time: float,
        best_algorithm: str,
        artifacts: Dict[str, str],
        selected_features: List[str] = None
    ) -> Dict[str, Any]:
        best_result = training_results[best_algorithm]
        best_metrics = best_result['metrics']

        report = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'model_used': best_algorithm,
            'best_parameters': best_result['params'],
            'dataset_info': {
                'total_samples': dataset_result['imbalance_info']['total_samples'],
                'train_samples': len(dataset_result['X_train']),
                'test_samples': len(dataset_result['X_test']),
                'num_features': len(dataset_result['feature_names']),
                'balancing_method': dataset_result['balancing_method'],
                'class_distribution': dataset_result['imbalance_info']['class_distribution']
            },
            'metrics_summary': {
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'f1_score': best_metrics['f1']
            },
            'class_performance': best_metrics['classification_report'],
            'confusion_matrix': best_metrics['confusion_matrix'],
            'model_comparison': {
                alg: {
                    'cv_score': res.get('cv_score', 0),
                    'test_f1': res.get('metrics', {}).get('f1', 0),
                    'test_accuracy': res.get('metrics', {}).get('accuracy', 0)
                }
                for alg, res in training_results.items()
                if 'error' not in res
            },
            'feature_selection': {
                'original_feature_count': len(dataset_result['feature_names']),
                'selected_feature_count': len(selected_features) if selected_features else 0,
                'selected_features': selected_features if selected_features else [],
                'feature_importance': training_results[best_algorithm].get('feature_importance', {})
            },
            'artifacts': artifacts
        }

        return report

    def _update_model_registry(
        self,
        service_conf_id: int,
        run_id: str,
        best_algorithm: str,
        best_metrics: Dict[str, Any],
        artifacts: Dict[str, str],
        model_uri: str
    ):
        try:
            existing_model = db_client.get_model_by_service_conf(
                service_conf_id)

            config_json = {
                'run_id': run_id,
                'algorithm': best_algorithm,
                'training_date': datetime.now().isoformat(),
                'confusion_matrix': best_metrics['confusion_matrix'],
                'classification_report': best_metrics['classification_report']
            }

            if existing_model:
                db_client.update_model_registry(
                    model_registry_id=existing_model['model_registry_id'],
                    name=f"malicious_url_detector_{best_algorithm}",
                    algorithm=best_algorithm,
                    mlflow_id=run_id,
                    model_uri=artifacts.get('model', ''),
                    scaler_uri=artifacts.get('scaler', ''),
                    label_uri=artifacts.get('label_encoder', ''),
                    accuracy_score=best_metrics['accuracy'],
                    recall_score=best_metrics['recall'],
                    precision_score=best_metrics['precision'],
                    f1_score=best_metrics['f1'],
                    config_json=config_json
                )
                logger.info(
                    f"Updated model registry for service_conf_id: {service_conf_id}")
            else:
                db_client.create_model_registry(
                    service_conf_id=service_conf_id,
                    name=f"malicious_url_detector_{best_algorithm}",
                    algorithm=best_algorithm,
                    mlflow_id=run_id,
                    model_uri=artifacts.get('model', ''),
                    scaler_uri=artifacts.get('scaler', ''),
                    label_uri=artifacts.get('label_encoder', ''),
                    accuracy_score=best_metrics['accuracy'],
                    recall_score=best_metrics['recall'],
                    precision_score=best_metrics['precision'],
                    f1_score=best_metrics['f1'],
                    config_json=config_json
                )
                logger.info(
                    f"Created model registry for service_conf_id: {service_conf_id}")

        except Exception as e:
            logger.error(f"Error updating model registry: {str(e)}")


training_service = TrainingService()
