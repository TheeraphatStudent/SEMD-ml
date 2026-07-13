from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from core import settings

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised when xgboost is unavailable
    XGBClassifier = None


@dataclass(frozen=True)
class ModelDefinition:
    identifier: str
    estimator_class: type
    default_params: Dict[str, Any]


class ModelFactory:
    def __init__(self) -> None:
        self.random_state = settings.random_state

    def available_models(self) -> Dict[str, ModelDefinition]:
        configs = settings.algorithm_hyperparameters
        models: Dict[str, ModelDefinition] = {
            "svm": ModelDefinition("svm", SVC, configs["svm"]),
            "random_forest": ModelDefinition(
                "random_forest",
                RandomForestClassifier,
                configs["random_forest"],
            ),
            "gradient_boosting": ModelDefinition(
                "gradient_boosting",
                GradientBoostingClassifier,
                configs["gradient_boosting"],
            ),
        }

        if XGBClassifier is not None and "xgboost" in configs:
            models["xgboost"] = ModelDefinition("xgboost", XGBClassifier, configs["xgboost"])

        return models

    def identifiers(self) -> list[str]:
        return list(self.available_models().keys())

    def build(self, identifier: str, overrides: Dict[str, Any] | None = None) -> Any:
        definitions = self.available_models()
        if identifier not in definitions:
            raise ValueError(f"Unknown algorithm: {identifier}")

        definition = definitions[identifier]
        params = dict(definition.default_params)
        if "random_state" not in params:
            params["random_state"] = self.random_state
        if overrides:
            params.update(overrides)
        return definition.estimator_class(**params)


model_factory = ModelFactory()
