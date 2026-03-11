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
from sklearn.metrics import classification_report

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

            logger.info(
                '\n- Step 4: Using all features (no feature selection)')
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
                X_sample=X_test_selected[:5] if len(
                    X_test_selected) > 0 else None,
                description=f"Trained {best_algorithm} model with F1: {best_metrics['f1']:.4f}"
            )
            model_uri = registration_result.get(
                'model_uri') if registration_result else None

            if registration_result and registration_result.get('version'):
                try:
                    mlflow_tracker.client.transition_model_version_stage(
                        name=model_name,
                        version=registration_result['version'],
                        stage="Production"
                    )
                    logger.info(f"Transitioned model {model_name} version {registration_result['version']} to Production stage")
                except Exception as e:
                    logger.warning(f"Failed to transition model to Production stage: {str(e)}")

            evaluation_result = mlflow_tracker.evaluate_model(
                model=ml_pipeline.best_model,
                X_test=X_test_selected,
                y_test=y_test_encoded,
                model_name=model_name
            )

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            mlflow_tracker.log_params({'training_time_seconds': training_time})

            logger.info('\n- Step 7: Generating reports for all algorithms...')
            reports = {}
            for alg in training_results:
                if 'error' not in training_results[alg]:
                    report = self._generate_report(
                        run_id=run_id,
                        training_results=training_results,
                        dataset_result=dataset_result,
                        training_time=training_time,
                        algorithm=alg,
                        artifacts=artifacts,
                        selected_features=selected_features if 'selected_features' in locals() else None
                    )

                    report_path = os.path.join(
                        self.reports_path, f"{alg}_training_report_{run_id}.json")
                    report_converted = self._convert_numpy_types(report)
                    with open(report_path, 'w') as f:
                        json.dump(report_converted, f, indent=2)

                    mlflow_tracker.log_artifact(report_path)
                    reports[alg] = report
                    logger.info(f"Report generated for {alg}")

            logger.info('\n- Step 8: Registering best model...')

            logger.info('\n- Step 9: Classification Report...')
            print(f"\n{'=' * 60}")
            print(f"  Classification Report — {best_algorithm}")
            print(f"{'=' * 60}")
            print(classification_report(
                y_test_encoded,
                ml_pipeline.best_model.predict(X_test_selected),
                target_names=ml_pipeline.label_encoder.classes_,
                zero_division=0
            ))
            print(f"{'=' * 60}\n")

            logger.info('\n- Step 10: Updating database...')
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

            return self._convert_numpy_types({
                'status': 'success',
                'run_id': run_id,
                'best_algorithm': best_algorithm,
                'metrics': best_metrics,
                'training_time': training_time,
                'reports': reports,
                'best_report': reports.get(best_algorithm),
                'artifacts': artifacts
            })

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
        algorithm: str,
        artifacts: Dict[str, str],
        selected_features: List[str] = None
    ) -> Dict[str, Any]:
        best_result = training_results[algorithm]
        best_metrics = best_result['metrics']

        feature_names = dataset_result.get('feature_names', [])

        feature_importance_raw = training_results[algorithm].get('feature_importance', {})
        mapped_importance = {}
        for key, imp in feature_importance_raw.items():
            if key.startswith('feature_'):
                try:
                    idx = int(key.split('_')[1])
                    if idx < len(feature_names):
                        mapped_importance[feature_names[idx]] = imp
                except (ValueError, IndexError):
                    mapped_importance[key] = imp
            else:
                mapped_importance[key] = imp

        report = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'model_used': algorithm,
            'best_parameters': best_result['params'],
            'dataset_info': {
                'total_samples': dataset_result['imbalance_info']['total_samples'],
                'train_samples': len(dataset_result['X_train']),
                'test_samples': len(dataset_result['X_test']),
                'num_features': len(dataset_result['feature_names']),
                'balancing_method': dataset_result['balancing_method'],
                'class_distribution': dataset_result['imbalance_info']['class_distribution']
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
                'feature_importance': mapped_importance
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


    def execute_training_obo(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting one-by-one training job: {job_data}")

        start_time = datetime.now()
        results = {
            'status': 'success',
            'datasets_trained': [],
            'datasets_failed': [],
            'models': {}
        }

        try:
            store_path = job_data.get('store_path')
            algorithms = job_data.get('algorithms', settings.available_algorithms)
            run_name_prefix = job_data.get('run_name', 'obo_training')
            balance_method = job_data.get('balance_method')

            if not store_path:
                raise ValueError('No store path provided')

            logger.info('\n' + '='*60)
            logger.info('STEP 1: Analyzing all datasets and creating benign_merge.csv...')
            logger.info('='*60)
            
            min_count_info = dataset_pipeline.calculate_min_class_count_across_datasets(store_path)
            dataset_stats = min_count_info['dataset_stats']
            benign_merge_path = min_count_info.get('benign_merge_path')

            all_dataset_names = list(dataset_stats.keys())
            
            logger.info(f"Total datasets to train: {len(all_dataset_names)}")
            logger.info(f"Benign merge path: {benign_merge_path}")

            results['min_count_info'] = self._convert_numpy_types(min_count_info)

            for idx, dataset_name in enumerate(all_dataset_names, 1):
                stats = dataset_stats[dataset_name]
                archive_info = stats['archive_info']
                clean_name = stats['clean_name']
                has_both_classes = stats['has_both_classes']
                
                logger.info('\n' + '='*60)
                logger.info(f'DATASET {idx}/{len(all_dataset_names)}: {dataset_name}')
                logger.info(f'Type: {"balanced" if has_both_classes else "single-class + benign_merge"}')
                logger.info('='*60)

                try:
                    if has_both_classes:
                        logger.info(f'\n- Preparing balanced dataset (using min class count)...')
                        dataset_result = dataset_pipeline.prepare_dataset_obo(
                            archive_info=archive_info,
                            apply_balancing=True
                        )
                    else:
                        logger.info(f'\n- Preparing single-class dataset with benign_merge...')
                        dataset_result = dataset_pipeline.prepare_dataset_single_class(
                            archive_info=archive_info,
                            benign_merge_path=benign_merge_path
                        )

                    X_train = dataset_result['X_train']
                    X_test = dataset_result['X_test']
                    y_train = dataset_result['y_train']
                    y_test = dataset_result['y_test']

                    run_name = f"{run_name_prefix}_{clean_name}"
                    run_id = mlflow_tracker.start_run(run_name=run_name)

                    mlflow_tracker.log_params({
                        'dataset_name': dataset_name,
                        'clean_name': clean_name,
                        'samples_per_class': dataset_result.get('samples_per_class', 0),
                        'original_size': dataset_result['original_size'],
                        'sampled_size': dataset_result['sampled_size'],
                        'training_mode': 'one_by_one',
                        'balancing_method': dataset_result.get('balancing_method', 'none')
                    })
                    mlflow_tracker.log_dataset_info(dataset_result)

                    logger.info('\n- Preprocessing data...')
                    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = ml_pipeline.preprocess_data(
                        X_train, X_test, y_train, y_test
                    )

                    logger.info(f'\n- Training with algorithms: {algorithms}...')
                    training_results = ml_pipeline.train_and_compare_models(
                        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded,
                        algorithms=algorithms
                    )

                    if 'error' in training_results:
                        raise ValueError(f"Training failed: {training_results['error']}")

                    mlflow_tracker.log_training_results(training_results)

                    logger.info('\n- Saving artifacts...')
                    artifacts = self._save_artifacts_obo(run_id, clean_name, dataset_name)

                    for artifact_path in artifacts.values():
                        mlflow_tracker.log_artifact(artifact_path)

                    best_algorithm = ml_pipeline.best_algorithm
                    best_metrics = training_results[best_algorithm]['metrics']

                    model_name = f"{best_algorithm}_{clean_name}_model"
                    registration_result = mlflow_tracker.register_model(
                        ml_pipeline.best_model,
                        model_name,
                        tags={
                            'algorithm': best_algorithm,
                            'dataset': dataset_name,
                            'f1_score': str(best_metrics['f1']),
                            'accuracy': str(best_metrics['accuracy']),
                            'training_mode': 'one_by_one'
                        },
                        alias='champion',
                        description=f"OBO trained {best_algorithm} model on {dataset_name} with F1: {best_metrics['f1']:.4f}"
                    )

                    report = self._generate_report(
                        run_id=run_id,
                        training_results=training_results,
                        dataset_result=dataset_result,
                        training_time=(datetime.now() - start_time).total_seconds(),
                        algorithm=best_algorithm,
                        artifacts=artifacts,
                        selected_features=dataset_result['feature_names']
                    )

                    report_path = os.path.join(
                        self.reports_path, f"{best_algorithm}_{clean_name}_report_{run_id}.json")
                    report_converted = self._convert_numpy_types(report)
                    with open(report_path, 'w') as f:
                        json.dump(report_converted, f, indent=2)
                    mlflow_tracker.log_artifact(report_path)

                    mlflow_tracker.end_run(status='FINISHED')

                    results['datasets_trained'].append(dataset_name)
                    results['models'][dataset_name] = {
                        'run_id': run_id,
                        'algorithm': best_algorithm,
                        'metrics': self._convert_numpy_types(best_metrics),
                        'artifacts': artifacts,
                        'clean_name': clean_name
                    }

                    logger.info(f'\n✓ Successfully trained model for {dataset_name}')
                    logger.info(f'  Best algorithm: {best_algorithm}')
                    logger.info(f'  F1 Score: {best_metrics["f1"]:.4f}')
                    logger.info(f'  Accuracy: {best_metrics["accuracy"]:.4f}')

                except Exception as e:
                    logger.error(f"Failed to train on dataset {dataset_name}: {str(e)}", exc_info=True)
                    results['datasets_failed'].append({
                        'dataset': dataset_name,
                        'error': str(e)
                    })
                    if mlflow_tracker.active_run:
                        mlflow_tracker.end_run(status='FAILED')

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            results['total_training_time'] = total_time
            results['num_trained'] = len(results['datasets_trained'])
            results['num_failed'] = len(results['datasets_failed'])

            logger.info('\n' + '='*60)
            logger.info('ONE-BY-ONE TRAINING COMPLETE')
            logger.info('='*60)
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Datasets trained: {len(results['datasets_trained'])}")
            logger.info(f"Datasets failed: {len(results['datasets_failed'])}")
            
            skipped = [name for name, stats in dataset_stats.items() 
                      if not stats.get('has_both_classes', True)]
            if skipped:
                results['skipped_datasets'] = skipped
                logger.info(f"Datasets skipped (single-class): {len(skipped)}")
                for name in skipped:
                    logger.info(f"  - {name}")

            if results['datasets_failed']:
                results['status'] = 'partial_success' if results['datasets_trained'] else 'failed'

            return self._convert_numpy_types(results)

        except Exception as e:
            logger.error(f"One-by-one training failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _save_artifacts_obo(self, run_id: str, clean_name: str, dataset_name: str) -> Dict[str, str]:
        base_models_path = os.path.join(os.path.dirname(self.reports_path), 'models')
        dataset_model_path = os.path.join(base_models_path, clean_name)
        os.makedirs(dataset_model_path, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        artifacts = {}

        # Save model
        model_filename = f"{ml_pipeline.best_algorithm}_{dataset_name}_model_{timestamp}_{run_id}.pkl"
        model_path = os.path.join(dataset_model_path, model_filename)
        import joblib
        joblib.dump(ml_pipeline.best_model, model_path)
        artifacts['model'] = model_path
        logger.info(f"Saved model to {model_path}")

        # Save scaler
        scaler_filename = f"scaler_{dataset_name}_{timestamp}_{run_id}.pkl"
        scaler_path = os.path.join(dataset_model_path, scaler_filename)
        joblib.dump(ml_pipeline.scaler, scaler_path)
        artifacts['scaler'] = scaler_path
        logger.info(f"Saved scaler to {scaler_path}")

        # Save label encoder
        label_encoder_filename = f"label_encoder_{dataset_name}_{timestamp}_{run_id}.pkl"
        label_encoder_path = os.path.join(dataset_model_path, label_encoder_filename)
        joblib.dump(ml_pipeline.label_encoder, label_encoder_path)
        artifacts['label_encoder'] = label_encoder_path
        logger.info(f"Saved label encoder to {label_encoder_path}")

        return artifacts


training_service = TrainingService()
