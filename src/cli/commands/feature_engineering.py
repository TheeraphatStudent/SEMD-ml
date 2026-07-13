from typing import Any

from core import get_logger, features_config
from features import feature_extractor

from ..common import emit_result

logger = get_logger(__name__)


def cmd_feature_engineering(args: Any) -> int:
    logger.info('Starting feature engineering analysis...')

    feature_groups = features_config.get_feature_groups_map()
    all_features = features_config.get_all_features()

    analysis = {
        'total_features': len(all_features),
        'feature_groups': {
            group: len(features)
            for group, features in feature_groups.items()
        },
        'class_emphasis': features_config.class_feature_emphasis,
        'enabled_groups': list(feature_extractor.enabled_groups),
        'configuration': {}
    }

    if args.url:
        logger.info(f"Extracting features from URL: {args.url}")
        features = feature_extractor.extract(args.url)
        analysis['sample_extraction'] = {
            'url': args.url,
            'features': features
        }

    emit_result(analysis, args.output)

    return 0
