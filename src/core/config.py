import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from scipy.stats import uniform, loguniform


class MLServiceSettings(BaseSettings):

    redis_host: str = Field(default='localhost', env='REDIS_HOST')
    redis_port: int = Field(default=6379, env='REDIS_PORT')
    redis_password: str = Field(default='', env='REDIS_PASSWORD')
    redis_db: int = Field(default=0, env='REDIS_DB')

    postgres_host: str = Field(default='localhost', env='POSTGRES_HOST')
    postgres_port: int = Field(default=5432, env='POSTGRES_PORT')
    postgres_user: str = Field(default='postgres', env='POSTGRES_USER')
    postgres_password: str = Field(default='', env='POSTGRES_PASSWORD')
    postgres_db: str = Field(default='semd_db', env='POSTGRES_DB')

    mlflow_tracking_uri: str = Field(
        default='http://localhost:5000', env='MLFLOW_TRACKING_URI')
    mlflow_experiment_name: str = Field(
        default='malicious_url_detection', env='MLFLOW_EXPERIMENT_NAME')

    cloudflare_api_token: str = Field(
        default='----------------------------', env='CLOUDFLARE_API_TOKEN')
    cloudflare_account_id: str = Field(
        default='----------------------------', env='CLOUDFLARE_ACCOUNT_ID')

    training_queue: str = 'ml_training_queue'
    prediction_queue: str = 'ml_prediction_queue'
    result_queue: str = 'ml_result_queue'

    features_config_path: str = './features/features.yaml'
    datadict_config_path: str = './data_dict.yaml'
    dataset_path: str = './dataset/raw'
    extraction_path: str = './dataset/extraction'
    models_path: str = '../models'
    reports_path: str = '../reports'

    random_state: int = 42
    test_size: float = 0.3
    cv_folds: int = 5

    enable_feature_importance: bool = Field(
        default=True, env='ENABLE_FEATURE_IMPORTANCE')

    enable_class_weighting: bool = Field(
        default=True, env='ENABLE_CLASS_WEIGHTING')
    class_weight_mode: str = Field(default='soft', env='CLASS_WEIGHT_MODE')

    log_level: str = Field(default='INFO', env='LOG_LEVEL')

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', None]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'min_child_weight': [1, 3, 5]
                }
            },
            'svm': {
                'model': SGDClassifier(random_state=self.random_state, loss='log_loss'),
                'params': {
                    'alpha': loguniform(1e-6, 1e-1),
                    # 'loss': ['hinge', 'log_loss', 'modified_huber'],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'learning_rate': ['optimal', 'constant', 'adaptive']
                }
            }
        }

    @property
    def available_algorithms(self) -> List[str]:
        return list(self.algorithm_configs.keys())

    @property
    def valid_balance_methods(self) -> List[str]:
        return ['none', 'smote', 'oversampling', 'undersampling']

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


class FeaturesConfig:

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def features(self) -> List[Dict[str, Any]]:
        return self.config.get('features', [])

    @property
    def feature_groups(self) -> Dict[str, Any]:
        return self.config.get('feature_groups', {})

    @property
    def class_feature_emphasis(self) -> Dict[str, Any]:
        return self.config.get('class_feature_emphasis', {})

    def get_all_features(self) -> List[str]:
        if self.features:
            return [feature['name'] for feature in self.features if isinstance(feature, dict) and 'name' in feature]

        features = []
        for group_name, group_data in self.feature_groups.items():
            if 'features' in group_data:
                for feature in group_data['features']:
                    if isinstance(feature, dict):
                        features.append(feature['name'])
                    else:
                        features.append(feature)
        return features

    def get_features_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata = {}
        for feature in self.features:
            if isinstance(feature, dict) and 'name' in feature:
                metadata[feature['name']] = {
                    'type': feature.get('type', 'unknown'),
                    'description': feature.get('description', '')
                }
        return metadata

    def get_feature_groups_map(self) -> Dict[str, List[str]]:
        if self.features:
            return {'all_features': self.get_all_features()}

        groups_map = {}
        for group_name, group_data in self.feature_groups.items():
            group_features = []
            if 'features' in group_data:
                for feature in group_data['features']:
                    if isinstance(feature, dict):
                        group_features.append(feature['name'])
                    else:
                        group_features.append(feature)
            groups_map[group_name] = group_features
        return groups_map

    def get_class_emphasis_features(self, class_name: str) -> List[str]:
        if class_name in self.class_feature_emphasis:
            return self.class_feature_emphasis[class_name].get('strong_features', [])
        return []

    def reload_config(self):
        self.config = self._load_config()


settings = MLServiceSettings()
features_config = FeaturesConfig(settings.features_config_path)
