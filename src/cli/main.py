import argparse
import sys

from cli import (
    cmd_data_migrate,
    cmd_data_migrate_feature,
    cmd_data_validate,
    cmd_evaluate,
    cmd_feature_engineering,
    cmd_predict,
    cmd_predict_test,
    cmd_promote_model,
    cmd_queue_status,
    cmd_register_model,
    cmd_rollback_model,
    cmd_train,
    cmd_train_obo,
    cmd_worker,
)
from core import get_logger, settings, setup_logging
from ml.model_factory import model_factory

setup_logging(settings.log_level)
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="SEMD ML Service")
    subparsers = parser.add_subparsers(dest="command")
    available_algorithms = model_factory.identifiers()
    algorithm_help = f"Available algorithms: {', '.join(available_algorithms)}"
    balance_help = f"Available balance methods: {', '.join(settings.valid_balance_methods)}"

    data_parser = subparsers.add_parser("data", help="Dataset operations")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    data_validate = data_subparsers.add_parser("validate", help="Validate training data")
    data_validate.add_argument("--dataset-files", nargs="+", help="Dataset files to load")
    data_validate.add_argument("--output", "-o", help="Output file for results")

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--dataset-files", nargs="+", help="Dataset files to use")
    train_parser.add_argument("--service-conf-id", type=int, help="Service configuration ID")
    train_parser.add_argument("--model", help=algorithm_help)
    train_parser.add_argument("--algorithms", nargs="+", help=algorithm_help)
    train_parser.add_argument("--balance", choices=settings.valid_balance_methods, help=balance_help)
    train_parser.add_argument("--run-name", help="Custom run name")
    train_parser.add_argument("--output", "-o", help="Output file for results")

    train_obo_parser = subparsers.add_parser("train-obo", help="Legacy one-by-one training")
    train_obo_parser.add_argument("--store-path", help="Path to dataset store directory")
    train_obo_parser.add_argument("--model", help=algorithm_help)
    train_obo_parser.add_argument("--algorithms", nargs="+", help=algorithm_help)
    train_obo_parser.add_argument("--balance", choices=settings.valid_balance_methods, help=balance_help)
    train_obo_parser.add_argument("--run-name", help="Custom run name")
    train_obo_parser.add_argument("--output", "-o", help="Output file for results")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    evaluate_parser.add_argument("--dataset-files", nargs="+", help="Dataset files to use")
    evaluate_parser.add_argument("--model", help=algorithm_help)
    evaluate_parser.add_argument("--algorithms", nargs="+", help=algorithm_help)
    evaluate_parser.add_argument("--balance", choices=settings.valid_balance_methods, help=balance_help)
    evaluate_parser.add_argument("--run-name", help="Custom run name")
    evaluate_parser.add_argument("--output", "-o", help="Output file for results")

    predict_parser = subparsers.add_parser("predict", help="Predict URL classification")
    predict_parser.add_argument("url", nargs="?", help="URL to predict")
    predict_parser.add_argument("--urls", nargs="+", help="Multiple URLs to predict")
    predict_parser.add_argument("--url-file", help="File containing URLs")
    predict_parser.add_argument("--model-id", help="Model artifact path or identifier")
    predict_parser.add_argument("--output", "-o", help="Output file for results")

    register_parser = subparsers.add_parser("register", help="Register an MLflow run as a candidate model")
    register_parser.add_argument("--run-id", required=True, help="MLflow run ID to register")
    register_parser.add_argument("--output", "-o", help="Output file for results")

    promote_parser = subparsers.add_parser("promote", help="Validate and promote a model version to champion")
    promote_parser.add_argument("--model-version", help="Explicit MLflow model version to validate and promote")
    promote_parser.add_argument("--output", "-o", help="Output file for results")

    rollback_parser = subparsers.add_parser("rollback", help="Rollback previous-champion to champion")
    rollback_parser.add_argument("--output", "-o", help="Output file for results")

    predict_test_parser = subparsers.add_parser("predict-test", help="Batch prediction")
    predict_test_parser.add_argument("--url", help="Single URL to test")
    predict_test_parser.add_argument("--urls", nargs="+", help="Multiple URLs to test")
    predict_test_parser.add_argument("--csv", help="CSV file with URLs")
    predict_test_parser.add_argument("--model-id", help="Model artifact path or identifier")
    predict_test_parser.add_argument("--output", "-o", help="Output file for results")

    feature_parser = subparsers.add_parser("feature-engineering", help="Analyze feature engineering")
    feature_parser.add_argument("--url", help="Sample URL to extract features from")
    feature_parser.add_argument("--output", "-o", help="Output file for analysis")

    worker_parser = subparsers.add_parser("worker", help="Start queue worker")
    worker_parser.add_argument("--mode", choices=["training", "prediction", "combined"], default="combined")

    subparsers.add_parser("queue-status", help="Show Redis queue status")

    migrate_parser = subparsers.add_parser("data-migrate", help="Extract datasets from archives")
    migrate_parser.add_argument("--store-path", help="Path to store directory containing archives")
    migrate_parser.add_argument("--raw-path", help="Path to raw directory for extracted CSV files")
    migrate_parser.add_argument("--output", "-o", help="Output file for migration report")

    feature_migrate_parser = subparsers.add_parser("data-migrate-feature", help="Migrate feature datasets")
    feature_migrate_parser.add_argument("--store-path", help="Path to store directory containing feature CSV files")
    feature_migrate_parser.add_argument("--raw-path", help="Path to raw directory for migrated CSV files")
    feature_migrate_parser.add_argument("--config", help="Path to dataset_feature.yaml config file")
    feature_migrate_parser.add_argument("--output", "-o", help="Output file for migration report")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "data" and args.data_command == "validate":
            return cmd_data_validate(args)
        if args.command == "train":
            return cmd_train(args)
        if args.command == "train-obo":
            return cmd_train_obo(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
        if args.command == "predict":
            return cmd_predict(args)
        if args.command == "register":
            return cmd_register_model(args)
        if args.command == "promote":
            return cmd_promote_model(args)
        if args.command == "rollback":
            return cmd_rollback_model(args)
        if args.command == "predict-test":
            return cmd_predict_test(args)
        if args.command == "feature-engineering":
            return cmd_feature_engineering(args)
        if args.command == "worker":
            return cmd_worker(args)
        if args.command == "queue-status":
            return cmd_queue_status(args)
        if args.command == "data-migrate":
            return cmd_data_migrate(args)
        if args.command == "data-migrate-feature":
            return cmd_data_migrate_feature(args)
        parser.print_help()
        return 1
    except Exception as exc:
        logger.error("Command failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
