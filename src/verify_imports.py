import sys


class ImportVerifier:

    def __init__(self):
        self.test_results = []
        self.total_tests = 10
        self.passed_tests = 0

    def print_header(self):
        print('=' * 60)
        print('SEMD ML Service - Import Verification')
        print('=' * 60)

    def test_core_imports(self):
        print('\n[1/10] Testing core imports...')
        from core import settings, features_config, get_logger, setup_logging
        print('✓ Core imports successful')
        self.passed_tests += 1
        return settings, features_config

    def test_features_imports(self):
        print('\n[2/10] Testing features imports...')
        from features import feature_extractor, FeatureExtractor
        print('✓ Features imports successful')
        self.passed_tests += 1
        return feature_extractor

    def test_data_imports(self):
        print('\n[3/10] Testing data imports...')
        from data import dataset_pipeline, DatasetPipeline
        print('✓ Data imports successful')
        self.passed_tests += 1

    def test_ml_imports(self):
        print('\n[4/10] Testing ml imports...')
        from ml import ml_pipeline, MLPipeline
        from ml import training_service, TrainingService
        from ml import prediction_service, PredictionService
        print('✓ ML imports successful')
        self.passed_tests += 1

    def test_infra_imports(self):
        print('\n[5/10] Testing infra imports...')
        from infra import db_client, DatabaseClient
        from infra import redis_client, RedisClient
        print('✓ Infrastructure imports successful')
        self.passed_tests += 1

    def test_tracking_imports(self):
        print('\n[6/10] Testing tracking imports...')
        from tracking import mlflow_tracker, MLflowTracker
        print('✓ Tracking imports successful')
        self.passed_tests += 1

    def test_worker_imports(self):
        print('\n[7/10] Testing worker imports...')
        from workers import QueueWorker
        print('✓ Worker imports successful')
        self.passed_tests += 1

    def test_cli_imports(self):
        print('\n[8/10] Testing CLI imports...')
        from cli import cmd_train, cmd_predict, cmd_evaluate
        print('✓ CLI imports successful')
        self.passed_tests += 1

    def test_backward_compatibility(self):
        print('\n[9/10] Testing backward compatibility...')
        from core.config import settings as old_settings
        from features.feature_extractor import FeatureExtractor as old_extractor
        from ml.ml_pipeline import MLPipeline as old_pipeline
        print('✓ Backward compatibility successful')
        self.passed_tests += 1

    def test_feature_extraction(self, feature_extractor):
        print('\n[10/10] Testing feature extraction...')
        test_url = 'https://example.com/test'
        features = feature_extractor.extract(test_url)
        assert 'url_length' in features
        assert features['url_length'] == len(test_url)
        print(
            f"✓ Feature extraction working (extracted {len(features)} features)")
        self.passed_tests += 1

    def print_summary(self):
        print('\n' + '=' * 60)
        print('✅ ALL IMPORTS VERIFIED SUCCESSFULLY')
        print('=' * 60)
        print(f"\nTests passed: {self.passed_tests}/{self.total_tests}")
        print('\nProject structure:')
        print('  ✓ Modular architecture')
        print('  ✓ Clean imports')
        print('  ✓ Backward compatibility')
        print('  ✓ Feature extraction functional')
        print('\nReady for:')
        print('  • CLI execution: cd src && python main.py <command>')
        print('  • Queue workers: cd src && python main.py worker')
        print('  • Verification: cd src && python verify_imports.py')
        print('  • Training SVM: cd src && python main.py train --dataset-files dataset/file.csv --algorithms svm')

    def run_all_tests(self):
        try:
            self.print_header()

            settings, features_config = self.test_core_imports()
            feature_extractor = self.test_features_imports()
            self.test_data_imports()
            self.test_ml_imports()
            self.test_infra_imports()
            self.test_tracking_imports()
            self.test_worker_imports()
            self.test_cli_imports()
            self.test_backward_compatibility()
            self.test_feature_extraction(feature_extractor)

            self.print_summary()
            return 0

        except ImportError as e:
            print(f"\n❌ Import failed: {e}")
            print('\nPlease check:')
            print('  1. All __init__.py files are present')
            print('  2. Import paths are correct')
            print('  3. No circular imports exist')
            return 1
        except Exception as e:
            print(f"\n❌ Verification failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    verifier = ImportVerifier()
    sys.exit(verifier.run_all_tests())
