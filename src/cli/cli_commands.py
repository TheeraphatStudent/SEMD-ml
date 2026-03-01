import json
import pandas as pd
from typing import Any
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from workers import QueueWorker
from queues import QueueManager
from core import get_logger, settings
from ml import training_service, prediction_service, ml_pipeline
from data import dataset_pipeline
from features import feature_extractor
from core import features_config
from infra import redis_client
import os
import sys

logger = get_logger(__name__)


def validate_algorithms(algorithms: list) -> bool:
    if not algorithms:
        return True

    available_algorithms = list(ml_pipeline.get_algorithm_configs().keys())
    invalid_algorithms = [
        alg for alg in algorithms if alg not in available_algorithms]

    if invalid_algorithms:
        logger.error(f"Invalid algorithm(s): {', '.join(invalid_algorithms)}")
        logger.error(
            f"Available algorithms: {', '.join(available_algorithms)}")
        return False

    return True


def display_comparison_table(results):
    if not results or len(results) < 2:
        return

    print('\nComparison table:')
    print('-' * 50)
    for i, result in enumerate(results):
        if result.get('status') == 'success':
            url = result.get('url', f'URL {i+1}')
            prediction = result.get('prediction', {})
            pred_class = prediction.get('class', 'unknown')
            confidence = prediction.get('confidence', 0)
            short_url = url[:30] + '...' if len(url) > 33 else url
            print(f"{i+1}) {short_url:<32} | {pred_class:<10} | {confidence:.4f}")
        else:
            url = result.get('url', f'URL {i+1}')
            short_url = url[:30] + '...' if len(url) > 33 else url
            print(f"{i+1}) {short_url:<32} | {'ERROR':<10} | {'N/A':<6}")

    print('-' * 50)

    all_features = set()
    for result in results:
        if result.get('status') == 'success' and 'features' in result:
            all_features.update(result['features'].keys())

    all_features = sorted(all_features)

    if not all_features:
        print('No features available for comparison')
        return

    header = 'Feature'.ljust(50) + ' | '
    for i, result in enumerate(results):
        if result.get('status') == 'success':
            prediction = result.get('prediction', {})
            pred_class = prediction.get('class', 'unknown')
            header += f"{i+1}) {pred_class:<23} | "
        else:
            header += f"{i+1}  {'ERROR':<23} | "
    header = header.rstrip(' | ')

    print(f"\n{header}")
    print('-' * len(header))

    for feature_name in all_features:
        row = f"{feature_name:<50} | "

        for i, result in enumerate(results):
            if result.get('status') == 'success' and 'features' in result:
                feature_value = result['features'].get(feature_name, 'N/A')
                if isinstance(feature_value, float):
                    value_str = f"{feature_value:.4f}"
                else:
                    value_str = str(feature_value)
                row += f"{value_str:<23} | "
            else:
                row += f"{'ERROR':<23} | "

        row = row.rstrip(' | ')
        print(row)

    print('=' * len(header))


def cmd_train(args: Any) -> int:
    logger.info('Starting training from CLI...')

    algorithms = args.algorithms or settings.available_algorithms
    if not validate_algorithms(algorithms):
        return 1

    job_data = {
        'service_conf_id': args.service_conf_id,
        'dataset_files': args.dataset_files,
        'algorithms': algorithms,
        'run_name': args.run_name,
        'balance_method': getattr(args, 'balance', None)
    }

    result = training_service.execute_training(job_data)

    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.json'):
            output_filename += '.json'

        reports_dir = settings.reports_path
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        print(json.dumps(result, indent=2))

    return 0 if result['status'] == 'success' else 1


def cmd_predict(args: Any) -> int:
    logger.info('Starting prediction from CLI...')

    if args.model_id:
        prediction_service.load_model(args.model_id)

    if args.url:
        urls = [u.strip() for u in args.url.split(',') if u.strip()]
    elif args.urls:
        urls = args.urls
    elif args.url_file:
        with open(args.url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        logger.error('Must provide --url, --urls, or --url-file')
        return 1

    if args.compare:
        if len(urls) < 2:
            logger.error('--compare requires at least 2 URLs')
            return 1
        if len(urls) > 5:
            logger.error('--compare supports maximum 5 URLs')
            return 1
        logger.info(f'Running comparison mode with {len(urls)} URLs')

    results = []
    for url in urls:
        job_data = {
            'url': url,
            'user_id': args.user_id,
            'model_id': args.model_id,
            'compare': args.compare
        }
        result = prediction_service.execute_prediction(job_data)
        results.append(result)

    if args.compare and len(results) > 1:
        display_comparison_table(results)

    output_data = {'predictions': results}

    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.json'):
            output_filename += '.json'

        reports_dir = settings.reports_path
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        print(json.dumps(output_data, indent=2))

    return 0


def cmd_evaluate(args: Any) -> int:
    logger.info('Starting evaluation from CLI...')

    algorithms = args.algorithms or settings.available_algorithms
    if not validate_algorithms(algorithms):
        return 1

    logger.info(f"Loading dataset from {args.dataset_files}")

    balance_method = getattr(args, 'balance', None)
    apply_balancing = not args.no_balancing

    dataset_result = dataset_pipeline.prepare_dataset(
        dataset_files=args.dataset_files,
        apply_balancing=apply_balancing,
        manual_balance_method=balance_method
    )

    X_train = dataset_result['X_train']
    X_test = dataset_result['X_test']
    y_train = dataset_result['y_train']
    y_test = dataset_result['y_test']

    logger.info('Preprocessing data...')
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = ml_pipeline.preprocess_data(
        X_train, X_test, y_train, y_test
    )

    logger.info('Performing feature selection...')
    X_train_df = pd.DataFrame(
        X_train_scaled, columns=dataset_result['feature_names'])

    X_train_selected, selected_features = ml_pipeline.feature_selection(
        X_train_df, y_train_encoded, dataset_result['feature_names']
    )

    logger.info('Training models...')
    results = ml_pipeline.train_and_compare_models(
        X_train_selected, X_test_scaled, y_train_encoded, y_test_encoded,
        algorithms=algorithms
    )

    evaluation_report = {
        'dataset_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(dataset_result['feature_names']),
            'selected_features': len(selected_features)
        },
        'results': {
            alg: {
                'metrics': res['metrics'],
                'cv_score': res['cv_score'],
                'params': res['params']
            }
            for alg, res in results.items()
            if 'error' not in res
        },
        'best_algorithm': ml_pipeline.best_algorithm
    }

    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.json'):
            output_filename += '.json'

        reports_dir = settings.reports_path
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")
    else:
        print(json.dumps(evaluation_report, indent=2))

    return 0


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
        'configuration': {
            'correlation_filter': settings.enable_correlation_filter,
            'variance_threshold': settings.enable_variance_threshold,
            'mutual_information': settings.enable_mutual_information,
            'feature_selection_k': settings.feature_selection_k
        }
    }

    if args.url:
        logger.info(f"Extracting features from URL: {args.url}")
        features = feature_extractor.extract(args.url)
        analysis['sample_extraction'] = {
            'url': args.url,
            'features': features
        }

    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.json'):
            output_filename += '.json'

        reports_dir = settings.reports_path
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Feature analysis saved to {output_path}")
    else:
        print(json.dumps(analysis, indent=2))

    return 0


def cmd_worker(args: Any) -> int:
    logger.info('Starting queue worker...')

    worker = QueueWorker()

    try:
        if args.mode == 'training':
            worker.start_training_worker()
        elif args.mode == 'prediction':
            worker.start_prediction_worker()
        else:
            worker.start_combined_worker()
    except KeyboardInterrupt:
        logger.info('Worker interrupted by user')
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
        return 1


def cmd_queue_status(args: Any) -> int:
    logger.info('Checking Redis queue status...')

    try:
        manager = QueueManager()
        status = manager.get_queue_status()
        manager.print_queue_status(status)
        return 0
    except Exception as e:
        logger.error(f"Failed to check queue status: {str(e)}")
        return 1


def cmd_data_migrate(args: Any) -> int:
    logger.info('Starting data migration from CLI...')

    store_path = Path(args.store_path) if args.store_path else Path(
        settings.dataset_path).parent / 'store'
    raw_path = Path(args.raw_path) if args.raw_path else Path(
        settings.dataset_path)

    if not store_path.exists():
        logger.error(f"Store path does not exist: {store_path}")
        return 1

    os.makedirs(raw_path, exist_ok=True)
    logger.info(f"Extracting datasets from {store_path} to {raw_path}")

    extracted_files = []
    processed_archives = []

    archive_extensions = ['.zip', '.tar.gz', '.tgz', '.tar', '.gz']
    archive_files = []

    for ext in archive_extensions:
        if ext == '.tar.gz':
            archive_files.extend(list(store_path.glob('*.tar.gz')))
        else:
            archive_files.extend(list(store_path.glob(f'*{ext}')))

    if not archive_files:
        logger.warning(f"No archive files found in {store_path}")
        return 1

    logger.info(f"Found {len(archive_files)} archive file(s) to process")

    for archive_file in archive_files:
        logger.info(f"Processing archive: {archive_file.name}")
        temp_extract_dir = raw_path / f'_temp_{archive_file.stem}'

        try:
            if archive_file.suffix == '.zip':
                with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                    logger.info(f"Extracted ZIP archive: {archive_file.name}")

            elif archive_file.name.endswith('.tar.gz') or archive_file.name.endswith('.tgz'):
                with tarfile.open(archive_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(temp_extract_dir)
                    logger.info(
                        f"Extracted TAR.GZ archive: {archive_file.name}")

            elif archive_file.suffix == '.tar':
                with tarfile.open(archive_file, 'r') as tar_ref:
                    tar_ref.extractall(temp_extract_dir)
                    logger.info(f"Extracted TAR archive: {archive_file.name}")

            elif archive_file.suffix == '.gz':
                output_file = temp_extract_dir / archive_file.stem
                os.makedirs(temp_extract_dir, exist_ok=True)
                with gzip.open(archive_file, 'rb') as gz_ref:
                    with open(output_file, 'wb') as out_ref:
                        shutil.copyfileobj(gz_ref, out_ref)
                logger.info(f"Extracted GZ archive: {archive_file.name}")

            csv_files = list(temp_extract_dir.rglob('*.csv'))

            if not csv_files:
                logger.warning(
                    f"No CSV files found in archive: {archive_file.name}")
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                continue

            logger.info(f"Found {len(csv_files)} CSV file(s) in archive")

            for csv_file in csv_files:
                dest_file = raw_path / csv_file.name

                if dest_file.exists():
                    base_name = csv_file.stem
                    counter = 1
                    while dest_file.exists():
                        dest_file = raw_path / f"{base_name}_{counter}.csv"
                        counter += 1

                shutil.move(str(csv_file), str(dest_file))
                extracted_files.append(dest_file.name)
                logger.info(
                    f"Moved CSV file: {csv_file.name} -> {dest_file.name}")

            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            processed_archives.append(archive_file.name)

        except Exception as e:
            logger.error(
                f"Error processing archive {archive_file.name}: {str(e)}")
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            continue

    migration_report = {
        'status': 'success' if extracted_files else 'no_files_extracted',
        'store_path': str(store_path),
        'raw_path': str(raw_path),
        'processed_archives': processed_archives,
        'extracted_files': extracted_files,
        'total_archives': len(processed_archives),
        'total_csv_files': len(extracted_files)
    }

    logger.info(
        f"Data migration complete: {len(extracted_files)} CSV file(s) extracted from {len(processed_archives)} archive(s)")

    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.json'):
            output_filename += '.json'

        reports_dir = settings.reports_path
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(migration_report, f, indent=2)
        logger.info(f"Migration report saved to {output_path}")
    else:
        print(json.dumps(migration_report, indent=2))

    return 0 if extracted_files else 1
