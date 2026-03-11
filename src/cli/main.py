import sys
import argparse

from core import setup_logging, get_logger, settings
from cli import (
    cmd_train,
    cmd_train_obo,
    cmd_predict,
    cmd_predict_test,
    cmd_evaluate,
    cmd_feature_engineering,
    cmd_worker,
    cmd_queue_status,
    cmd_data_migrate,
    cmd_data_migrate_feature
)
from ml import ml_pipeline

setup_logging(settings.log_level)
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='SEMD ML Service - Malicious URL Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument(
        '--dataset-files', nargs='+', required=True, help='Dataset files to use')
    train_parser.add_argument(
        '--service-conf-id', type=int, help='Service configuration ID')

    available_algorithms = list(ml_pipeline.get_algorithm_configs().keys())
    algorithm_help = f'Algorithms to train (available: {', '.join(available_algorithms)})'
    train_parser.add_argument('--algorithms', nargs='+', help=algorithm_help)

    valid_balance_methods = settings.valid_balance_methods
    balance_help = f'Manual balance method (available: {', '.join(valid_balance_methods)}). If not specified, auto-selection is used.'
    train_parser.add_argument(
        '--balance', choices=valid_balance_methods, help=balance_help)

    train_parser.add_argument('--run-name', help='Custom run name')
    train_parser.add_argument('--output', '-o', help='Output file for results')

    train_obo_parser = subparsers.add_parser(
        'train-obo', help='Train models one-by-one for each dataset in store')
    train_obo_parser.add_argument(
        '--store-path', help='Path to store directory containing dataset archives (default: dataset/store)')
    train_obo_parser.add_argument(
        '--algorithms', nargs='+', help=algorithm_help)
    train_obo_parser.add_argument(
        '--balance', choices=valid_balance_methods, help=balance_help)
    train_obo_parser.add_argument('--run-name', help='Custom run name prefix')
    train_obo_parser.add_argument('--output', '-o', help='Output file for results')

    predict_parser = subparsers.add_parser(
        'predict', help='Predict URL classification')
    predict_parser.add_argument('--url', help='Single URL to predict')
    predict_parser.add_argument(
        '--urls', nargs='+', help='Multiple URLs to predict (space-separated)')
    predict_parser.add_argument(
        '--url-file', help='File containing URLs (one per line)')
    predict_parser.add_argument('--model-id', help='Model ID to use')
    predict_parser.add_argument('--user-id', type=int, help='User ID')
    predict_parser.add_argument(
        '--compare', action='store_true',
        help='Show feature comparison table (requires 2-5 URLs)')
    predict_parser.add_argument(
        '--output', '-o', help='Output file for results')

    predict_test_parser = subparsers.add_parser(
        'predict-test', help='Batch test URLs with detailed metrics and timing')
    predict_test_parser.add_argument(
        '--url', help='Single URL to test')
    predict_test_parser.add_argument(
        '--urls', nargs='+', help='Multiple URLs to test (space-separated)')
    predict_test_parser.add_argument(
        '--csv', help='Path to CSV file with URLs (in dataset/test or full path)')
    predict_test_parser.add_argument(
        '--model-id', help='Model ID to use')
    predict_test_parser.add_argument(
        '--output', '-o', help='Output file for results (JSON)')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--dataset-files', nargs='+',
                             required=True, help='Dataset files to use')

    eval_algorithm_help = f'Algorithms to evaluate (available: {', '.join(available_algorithms)})'
    eval_parser.add_argument('--algorithms', nargs='+',
                             help=eval_algorithm_help)

    eval_parser.add_argument(
        '--balance', choices=valid_balance_methods, help=balance_help)
    eval_parser.add_argument(
        '--no-balancing', action='store_true', help='Disable dataset balancing')
    eval_parser.add_argument('--output', '-o', help='Output file for results')

    feature_parser = subparsers.add_parser(
        'feature-engineering', help='Analyze feature engineering')
    feature_parser.add_argument(
        '--url', help='Sample URL to extract features from')
    feature_parser.add_argument(
        '--output', '-o', help='Output file for analysis')

    worker_parser = subparsers.add_parser('worker', help='Start queue worker')
    worker_parser.add_argument(
        '--mode', choices=['training', 'prediction', 'combined'], default='combined')

    queue_status_parser = subparsers.add_parser(
        'queue-status', help='Show status of Redis queues')

    migrate_parser = subparsers.add_parser(
        'data-migrate', help='Extract datasets from archives to raw directory')
    migrate_parser.add_argument(
        '--store-path', help='Path to store directory containing archives (default: dataset/store)')
    migrate_parser.add_argument(
        '--raw-path', help='Path to raw directory for extracted CSV files (default: dataset/raw)')
    migrate_parser.add_argument(
        '--output', '-o', help='Output file for migration report')

    feature_migrate_parser = subparsers.add_parser(
        'data-migrate-feature', help='Migrate feature datasets from store to raw with column mapping')
    feature_migrate_parser.add_argument(
        '--store-path', help='Path to store directory containing feature CSV files (default: dataset/feature/store)')
    feature_migrate_parser.add_argument(
        '--raw-path', help='Path to raw directory for migrated CSV files (default: dataset/feature/raw)')
    feature_migrate_parser.add_argument(
        '--config', help='Path to dataset_feature.yaml config file (default: dataset/feature/dataset_feature.yaml)')
    feature_migrate_parser.add_argument(
        '--output', '-o', help='Output file for migration report')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'train':
            return cmd_train(args)
        elif args.command == 'train-obo':
            return cmd_train_obo(args)
        elif args.command == 'predict':
            return cmd_predict(args)
        elif args.command == 'predict-test':
            return cmd_predict_test(args)
        elif args.command == 'evaluate':
            return cmd_evaluate(args)
        elif args.command == 'feature-engineering':
            return cmd_feature_engineering(args)
        elif args.command == 'worker':
            return cmd_worker(args)
        elif args.command == 'queue-status':
            return cmd_queue_status(args)
        elif args.command == 'data-migrate':
            return cmd_data_migrate(args)
        elif args.command == 'data-migrate-feature':
            return cmd_data_migrate_feature(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.error(f"Command failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
