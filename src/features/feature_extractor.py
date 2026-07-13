from typing import Dict, List, Optional, Set

from core import features_config
from semd_ml.features.extractor import URLFeatureExtractor


class FeatureExtractor:
    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        enabled_groups: Optional[Set[str]] = None,
    ):
        self._extractor = URLFeatureExtractor(
            features_config=features_config,
            feature_weights=feature_weights,
            enabled_groups=enabled_groups,
        )
        self.features_list = self._extractor.features_list
        self.feature_groups_map = self._extractor.feature_groups_map
        self.feature_groups_config = self._extractor.feature_groups_config
        self.enabled_groups = self._extractor.enabled_groups
        self.schema = self._extractor.schema

    def extract(self, url: str, apply_weights: bool = False, target_class: Optional[str] = None):
        self._extractor.enabled_groups = self.enabled_groups
        return self._extractor.extract(url, apply_weights=apply_weights, target_class=target_class)

    def set_feature_weights(self, weights: Dict[str, float]):
        self._extractor.set_feature_weights(weights)

    def enable_feature_groups(self, groups: List[str]):
        self.enabled_groups = set(groups)
        self._extractor.enable_feature_groups(groups)

    def disable_feature_groups(self, groups: List[str]):
        self._extractor.disable_feature_groups(groups)
        self.enabled_groups = self._extractor.enabled_groups

    def get_enabled_features(self):
        self._extractor.enabled_groups = self.enabled_groups
        return list(self._extractor.get_enabled_features())


feature_extractor = FeatureExtractor()
