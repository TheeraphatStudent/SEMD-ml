import os
import joblib
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline

import logging

from core import settings, features_config
from scipy.stats import uniform, loguniform

from features import feature_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipeline:

    def __init__(self):
        self.random_state = settings.random_state
        self.cv_folds = settings.cv_folds
        self.models_path = settings.models_path

        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.variance_selector = None
        self.correlation_filter_mask = None
        self.mutual_info_selector = None
        self.feature_names = None
        self.feature_importance_scores = {}

        self.best_model = None
        self.best_algorithm = None
        self.best_params = None

        os.makedirs(self.models_path, exist_ok=True)

    def get_algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        return settings.algorithm_configs

    def preprocess_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info('Preprocessing data...')

        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(
            f"Data preprocessed: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")

        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

    def feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        logger.info(
            'Starting advanced multi-stage feature selection pipeline...')

        self.feature_names = feature_names
        X_current = X_train.copy()
        current_features = feature_names.copy()

        if settings.enable_correlation_filter:
            X_current, current_features = self._correlation_filter(
                X_current, current_features)

        if settings.enable_variance_threshold:
            X_current, current_features = self._variance_filter(
                X_current, current_features)

        if settings.enable_mutual_information:
            X_current, current_features = self._mutual_info_filter(
                X_current, y_train, current_features)

        if settings.enable_feature_selection:
            X_current, current_features = self._selectkbest_filter(
                X_current, y_train, current_features)

        logger.info(
            f"X shape before selection: {X_train.shape}")
        logger.info(
            f"X shape after selection: ({X_train.shape[0]}, {len(current_features)})")
        logger.info(
            f"Feature selection complete: {len(feature_names)} -> {len(current_features)} features")
        self.selected_feature_names = current_features

        if current_features:
            features_str = ', '.join(current_features)
            logger.info(
                f"Selected features ({len(current_features)}): {features_str}")
        else:
            logger.warning(
                'No features were selected after feature selection pipeline')

        return X_current.values, current_features

    def _correlation_filter(self, X: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(
            f"Applying correlation filter (threshold={settings.correlation_threshold})...")

        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper_triangle.columns if any(
            upper_triangle[column] > settings.correlation_threshold)]

        X_filtered = X.drop(columns=to_drop)
        remaining_features = [f for f in feature_names if f not in to_drop]

        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return X_filtered, remaining_features

    def _variance_filter(self, X: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(
            f"Applying variance threshold filter (threshold={settings.variance_threshold})...")

        self.variance_selector = VarianceThreshold(
            threshold=settings.variance_threshold)
        X_filtered = self.variance_selector.fit_transform(X)

        selected_mask = self.variance_selector.get_support()
        remaining_features = [f for f, selected in zip(
            feature_names, selected_mask) if selected]

        logger.info(
            f"Removed {len(feature_names) - len(remaining_features)} low variance features")
        return pd.DataFrame(X_filtered, columns=remaining_features), remaining_features

    def _mutual_info_filter(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(
            f"Applying mutual information filter (threshold={settings.mutual_info_threshold})...")

        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)

        selected_mask = mi_scores > settings.mutual_info_threshold
        remaining_features = [f for f, selected in zip(
            feature_names, selected_mask) if selected]

        if len(remaining_features) == 0:
            logger.warning(
                'No features passed mutual information filter, keeping all')
            return X, feature_names

        X_filtered = X[remaining_features]

        logger.info(
            f"Removed {len(feature_names) - len(remaining_features)} low mutual information features")
        return X_filtered, remaining_features

    def _selectkbest_filter(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        k = min(settings.feature_selection_k, len(feature_names))
        logger.info(f"Applying SelectKBest filter (k={k})...")

        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_filtered = self.feature_selector.fit_transform(X, y)

        selected_mask = self.feature_selector.get_support()
        remaining_features = [f for f, selected in zip(
            feature_names, selected_mask) if selected]

        logger.info(f"Selected top {len(remaining_features)} features")
        return pd.DataFrame(X_filtered, columns=remaining_features), remaining_features

    def train_model(
        self,
        algorithm: str,
        X_train: np.ndarray,
        x_test: np.ndarray,
        Y_train: np.ndarray,
        y_test: np.ndarray,
        n_iter: int = 20
    ) -> Tuple[Any, Dict[str, Any], float]:
        logger.info(f"Training {algorithm} model with Pipeline...")

        configs = self.get_algorithm_configs()

        if algorithm not in configs:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        config = configs[algorithm]
        model = config['model']
        param_distributions = config['params']

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        pipeline_params = {}
        for key, value in param_distributions.items():
            pipeline_params[f'classifier__{key}'] = value

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=pipeline_params,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring='f1_weighted',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=2
        )

        random_search.fit(X_train, Y_train)

        y_pred = random_search.predict(x_test)
        print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        results_df = (pd.DataFrame(random_search.cv_results_)).sort_values(
            by='rank_test_score', ascending=False)

        # logger.info(f"{algorithm} training complete. Best CV score: {best_score:.4f}")
        # logger.info(f"Best parameters: {best_params}")

        print(f"\n{'-=' * 15} Summarize best {'=-' * 15}\n")
        # print(best_model)
        print(f"Test accuracy: {accuracy}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")
        print(results_df.head())
        print(f"\n{'-=' * 20}{'=-' * 20}\n")

        results_df.to_csv(
            f"{settings.reports_path}/{algorithm}_best-results.csv", index=False)

        return best_model, best_params, best_score

    
    # ---------------------------------------
    # ----- Extract feature importance ------
    # - ใช้เพื่อหา Feature ที่มีความสำคัญต่อ Algrolithm
    # ---------------------------------------

    def extract_feature_importance(self, model: Any, algorithm: str) -> Dict[str, float]:
        if not settings.enable_feature_importance:
            return {}

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).mean(axis=0) if len(
                    model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return {}

            if hasattr(self, 'selected_feature_names') and self.selected_feature_names:
                feature_names = self.selected_feature_names
            elif self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [
                    f"feature_{i}" for i in range(len(importances))]

            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

            logger.info(f"Extracted feature importance for {algorithm}")
            logger.info(
                f"Top 5 features: {list(sorted_importance.keys())[:5]}")

            return sorted_importance
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return {}

    # ---------------------------------------
    # ----------- Evaluate model ------------
    # ---------------------------------------

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        logger.info('Evaluating model...')

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(
            model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
        }

        logger.info(f"Evaluation metrics:")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - Precision: {metrics['precision']:.4f}")
        logger.info(f"  - Recall: {metrics['recall']:.4f}")
        logger.info(f"  - F1 Score: {metrics['f1']:.4f}")

        return metrics

    def train_and_compare_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if algorithms is None:
            algorithms = settings.available_algorithms

        logger.info(
            f"Training and comparing {len(algorithms)} algorithms: {algorithms}\n")

        results = {}

        for algorithm in algorithms:
            try:
                logger.info(f"Training {algorithm} model...")

                # ---------------------------------------
                # ------------- Train model -------------
                # ---------------------------------------

                model, params, cv_score = self.train_model(
                    algorithm=algorithm,
                    X_train=X_train,
                    x_test=X_test,
                    Y_train=y_train,
                    y_test=y_test
                )

                metrics = self.evaluate_model(
                    model=model, X_test=X_test, y_test=y_test)

                classifier = model.named_steps['classifier'] if hasattr(
                    model, 'named_steps') else model
                feature_importance = self.extract_feature_importance(
                    model=classifier, algorithm=algorithm)

                if feature_importance:
                    self.feature_importance_scores[algorithm] = feature_importance

                results[algorithm] = {
                    'model': model,
                    'params': params,
                    'cv_score': cv_score,
                    'metrics': metrics,
                    'feature_importance': feature_importance
                }

                logger.info(f"\n{'-=' * 20}-\n")

            except Exception as e:
                logger.error(f"Error training {algorithm}: {str(e)}")
                results[algorithm] = {
                    'error': str(e)
                }

                try:
                    from tracking import mlflow_tracker
                    mlflow_tracker.log_error(
                        error_message=str(e),
                        error_type=f"algorithm_{algorithm}_failure",
                        additional_info={'algorithm': algorithm}
                    )
                except ImportError:
                    pass

        successful_algorithms = [alg for alg in results if 'error' not in results[alg]]

        if not successful_algorithms:
            logger.error(f"No successful algorithms found.")
            logger.error(f"Algorithms attempted: {algorithms}")
            logger.error(f"Results: {results}")
            raise ValueError('All algorithms failed to train successfully')

        best_algorithm = None

        try:
            logger.debug(f"{'-=' * 20}-\n")
            logger.debug(results)

            best_algorithm = max(
                successful_algorithms,
                key=lambda alg: results[alg]['metrics']['f1']
            )

        except Exception as e:
            logger.error(f"No successful algorithms found. Error: {str(e)}")
            logger.error(f"Algorithms attempted: {algorithms}")
            logger.error(f"Results: {results}")

            return {
                'error': 'All algorithms failed to train successfully',
                'attempted_algorithms': algorithms,
                'individual_errors': {alg: res.get('error', 'Unknown error') for alg, res in results.items()}
            }

        self.best_model = results[best_algorithm]['model']
        self.best_algorithm = best_algorithm
        self.best_params = results[best_algorithm]['params']

        logger.info(
            f"Best algorithm: {best_algorithm} with F1 score: {results[best_algorithm]['metrics']['f1']:.4f}")

        return results

    def save_artifacts(self, run_id: str) -> Dict[str, str]:
        logger.info('Saving model artifacts...')

        prefix = f"{self.best_algorithm}_" if self.best_algorithm else ""
        artifacts = {}

        model_path = os.path.join(self.models_path, f"{prefix}model_{run_id}.pkl")
        joblib.dump(self.best_model, model_path)
        artifacts['model'] = model_path
        logger.info(f"Model saved to {model_path}")

        scaler_path = os.path.join(self.models_path, f"{prefix}scaler_{run_id}.pkl")
        joblib.dump(self.scaler, scaler_path)
        artifacts['scaler'] = scaler_path
        logger.info(f"Scaler saved to {scaler_path}")

        label_encoder_path = os.path.join(
            self.models_path, f"{prefix}label_encoder_{run_id}.pkl")
        joblib.dump(self.label_encoder, label_encoder_path)
        artifacts['label_encoder'] = label_encoder_path
        logger.info(f"Label encoder saved to {label_encoder_path}")

        if self.feature_selector:
            feature_selector_path = os.path.join(
                self.models_path, f"{prefix}feature_selector_{run_id}.pkl")
            joblib.dump(self.feature_selector, feature_selector_path)
            artifacts['feature_selector'] = feature_selector_path
            logger.info(f"Feature selector saved to {feature_selector_path}")

        if hasattr(self, 'selected_feature_names') and self.selected_feature_names:
            selected_features_path = os.path.join(
                self.models_path, f"{prefix}selected_features_{run_id}.json")
            with open(selected_features_path, 'w') as f:
                json.dump(self.selected_feature_names, f)
            artifacts['selected_features'] = selected_features_path
            logger.info(f"Selected features saved to {selected_features_path}")

        return artifacts

    def _find_artifact(self, run_id: str, artifact_type: str, ext: str = 'pkl') -> Optional[str]:
        import glob

        prefixed = glob.glob(os.path.join(self.models_path, f"*_{artifact_type}_{run_id}.{ext}"))
        if prefixed:
            return prefixed[0]

        fallback = os.path.join(self.models_path, f"{artifact_type}_{run_id}.{ext}")
        if os.path.exists(fallback):
            return fallback

        return None

    def load_artifacts(self, run_id: str) -> bool:
        try:
            model_path = self._find_artifact(run_id, 'model')
            if model_path is None:
                raise FileNotFoundError(f"Model artifact not found for run {run_id}")
            self.best_model = joblib.load(model_path)

            scaler_path = self._find_artifact(run_id, 'scaler')
            if scaler_path is None:
                raise FileNotFoundError(f"Scaler artifact not found for run {run_id}")
            self.scaler = joblib.load(scaler_path)

            label_encoder_path = self._find_artifact(run_id, 'label_encoder')
            if label_encoder_path is None:
                raise FileNotFoundError(f"Label encoder artifact not found for run {run_id}")
            self.label_encoder = joblib.load(label_encoder_path)

            feature_selector_path = self._find_artifact(run_id, 'feature_selector')
            if feature_selector_path:
                self.feature_selector = joblib.load(feature_selector_path)

            logger.info(f"Artifacts loaded for run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            return False

    def predict(self, url: str) -> Dict[str, Any]:
        if self.best_model is None:
            raise ValueError('No model loaded. Train or load a model first.')

        logger.info(f"Extracting features for URL: {url}")
        features = feature_extractor.extract(url)
        X = pd.DataFrame([features])

        logger.info(f"Total features extracted: {len(features)}")

        if self.feature_selector:
            all_feature_names = sorted(features.keys())
            selected_mask = self.feature_selector.get_support()
            
            if len(all_feature_names) != len(selected_mask):
                logger.warning(f"Feature count mismatch: extracted {len(all_feature_names)}, expected {len(selected_mask)}")
            
            selected_features = [name for name, selected in zip(
                all_feature_names, selected_mask) if selected]

            X_selected = X[selected_features]

            logger.info(
                f"Selected {len(selected_features)} features for prediction:")
            
            logger.info("-" * 68)
            logger.info(f"{'  Feature':<43} | {'Value':<15}")
            logger.info("-" * 68)
            
            for i, feature_name in enumerate(selected_features):
                feature_value = X_selected.iloc[0, i]
                if isinstance(feature_value, float):
                    value_str = f"{feature_value:.6f}"
                else:
                    value_str = str(feature_value)
                logger.info(f" - {feature_name:<40} | {value_str:<15}")

            X_final = X_selected.values
        else:
            X_final = X.values
            logger.info(
                f"Using all {X.shape[1]} features (no feature selection)")

        logger.info(f"Input shape for model: {X_final.shape}")

        prediction = self.best_model.predict(X_final)[0]
        prediction_proba = self.best_model.predict_proba(X_final)[0]

        logger.info(f"Model classes: {self.best_model.classes_}")
        logger.info(f"Raw prediction index: {prediction}")
        logger.info(f"Prediction probabilities: {prediction_proba}")

        predicted_class = self.label_encoder.inverse_transform([prediction])[0]

        class_probabilities = {
            cls: float(prob)
            for cls, prob in zip(self.label_encoder.classes_, prediction_proba)
        }

        logger.info(
            f"Final prediction: {predicted_class} with confidence {max(prediction_proba):.4f}")

        feature_dict = {}
        if self.feature_selector:
            selected_mask = self.feature_selector.get_support()
            all_feature_names = list(features.keys())
            selected_features = [name for name, selected in zip(
                all_feature_names, selected_mask) if selected]
            for feat_name in selected_features:
                feature_dict[feat_name] = features.get(feat_name)
        else:
            feature_dict = features

        return {
            'predicted_class': predicted_class,
            'confidence': float(max(prediction_proba)),
            'class_probabilities': class_probabilities,
            'features': feature_dict
        }


ml_pipeline = MLPipeline()
