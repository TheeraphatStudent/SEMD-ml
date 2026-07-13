import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


class MLServiceSettings(BaseSettings):
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")

    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="semd_db", env="POSTGRES_DB")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_container_tracking_uri: str = Field(
        default="http://mlflow:5000",
        env="MLFLOW_CONTAINER_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(default="semd-url-classification", env="MLFLOW_EXPERIMENT_NAME")
    mlflow_registered_model_name: str = Field(
        default="semd-malicious-url-detector",
        env="MLFLOW_REGISTERED_MODEL_NAME",
    )
    mlflow_artifact_root: str = Field(default="./artifacts/mlflow", env="MLFLOW_ARTIFACT_ROOT")
    mlflow_alias_candidate: str = Field(default="candidate", env="MLFLOW_ALIAS_CANDIDATE")
    mlflow_alias_champion: str = Field(default="champion", env="MLFLOW_ALIAS_CHAMPION")
    mlflow_alias_previous_champion: str = Field(
        default="previous-champion",
        env="MLFLOW_ALIAS_PREVIOUS_CHAMPION",
    )
    mlflow_local_fallback_enabled: bool = Field(default=False, env="MLFLOW_LOCAL_FALLBACK_ENABLED")
    mlflow_local_fallback_model_path: str = Field(default="", env="MLFLOW_LOCAL_FALLBACK_MODEL_PATH")
    mlflow_local_fallback_model_version: str = Field(default="", env="MLFLOW_LOCAL_FALLBACK_MODEL_VERSION")
    mlflow_local_fallback_model_name: str = Field(default="", env="MLFLOW_LOCAL_FALLBACK_MODEL_NAME")
    model_promotion_gates: str = Field(
        default=(
            '{"malicious_recall":{"operator":">=","threshold":0.95},'
            '"malicious_f1":{"operator":">=","threshold":0.93},'
            '"false_negative_rate":{"operator":"<=","threshold":0.05},'
            '"prediction_latency_ms":{"operator":"<=","threshold":200.0}}'
        ),
        env="MODEL_PROMOTION_GATES",
    )
    promotion_require_champion_comparison: bool = Field(
        default=True,
        env="PROMOTION_REQUIRE_CHAMPION_COMPARISON",
    )
    promotion_smoke_test_urls: str = Field(
        default='["https://example.com","http://secure-login.bad-example.net/verify"]',
        env="PROMOTION_SMOKE_TEST_URLS",
    )

    cloudflare_api_token: str = Field(default="----------------------------", env="CLOUDFLARE_API_TOKEN")
    cloudflare_account_id: str = Field(default="----------------------------", env="CLOUDFLARE_ACCOUNT_ID")

    training_queue: str = "ml_training_queue"
    prediction_queue: str = "ml_prediction_queue"
    result_queue: str = "ml_result_queue"

    features_config_path: str = str(PROJECT_ROOT / "src" / "features" / "features.yaml")
    datadict_config_path: str = "./data_dict.yaml"
    dataset_path: str = str(PROJECT_ROOT / "src" / "dataset" / "raw")
    extraction_path: str = str(PROJECT_ROOT / "src" / "dataset" / "extraction")
    models_path: str = str(PROJECT_ROOT / "models")
    reports_path: str = str(PROJECT_ROOT / "reports")

    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5

    default_dataset_files: List[str] = Field(default_factory=lambda: ["raw"])
    default_train_algorithms: List[str] = Field(
        default_factory=lambda: ["random_forest", "gradient_boosting"]
    )

    enable_feature_importance: bool = Field(default=True, env="ENABLE_FEATURE_IMPORTANCE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def algorithm_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        configs: Dict[str, Dict[str, Any]] = {
            "svm": {
                "C": 1.0,
                "kernel": "rbf",
                "probability": True,
                "class_weight": "balanced",
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "n_jobs": 1,
            },
            "gradient_boosting": {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 1.0,
            },
        }

        if _xgboost_available():
            configs["xgboost"] = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "n_jobs": 1,
            }

        return configs

    @property
    def available_algorithms(self) -> List[str]:
        return list(self.algorithm_hyperparameters.keys())

    @property
    def valid_balance_methods(self) -> List[str]:
        return ["none", "smote", "oversampling", "undersampling"]

    @property
    def parsed_model_promotion_gates(self) -> Dict[str, Dict[str, Any]]:
        try:
            payload = json.loads(self.model_promotion_gates)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid MODEL_PROMOTION_GATES JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("MODEL_PROMOTION_GATES must decode to an object")
        return payload

    @property
    def parsed_promotion_smoke_test_urls(self) -> List[str]:
        try:
            payload = json.loads(self.promotion_smoke_test_urls)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid PROMOTION_SMOKE_TEST_URLS JSON: {exc}") from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("PROMOTION_SMOKE_TEST_URLS must decode to a list of strings")
        return payload

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class FeaturesConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    @property
    def features(self) -> List[Dict[str, Any]]:
        return self.config.get("features", [])

    @property
    def feature_groups(self) -> Dict[str, Any]:
        return self.config.get("feature_groups", {})

    @property
    def class_feature_emphasis(self) -> Dict[str, Any]:
        return self.config.get("class_feature_emphasis", {})

    def get_all_features(self) -> List[str]:
        if self.features:
            return [feature["name"] for feature in self.features if isinstance(feature, dict) and "name" in feature]

        features = []
        for group_data in self.feature_groups.values():
            for feature in group_data.get("features", []):
                features.append(feature["name"] if isinstance(feature, dict) else feature)
        return features

    def get_features_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata = {}
        for feature in self.features:
            if isinstance(feature, dict) and "name" in feature:
                metadata[feature["name"]] = {
                    "type": feature.get("type", "unknown"),
                    "description": feature.get("description", ""),
                }
        return metadata

    def get_feature_groups_map(self) -> Dict[str, List[str]]:
        if self.features:
            return {"all_features": self.get_all_features()}

        groups_map = {}
        for group_name, group_data in self.feature_groups.items():
            features = []
            for feature in group_data.get("features", []):
                features.append(feature["name"] if isinstance(feature, dict) else feature)
            groups_map[group_name] = features
        return groups_map

    def get_class_emphasis_features(self, class_name: str) -> List[str]:
        if class_name in self.class_feature_emphasis:
            return self.class_feature_emphasis[class_name].get("strong_features", [])
        return []

    def reload_config(self) -> None:
        self.config = self._load_config()


settings = MLServiceSettings()
features_config = FeaturesConfig(settings.features_config_path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
