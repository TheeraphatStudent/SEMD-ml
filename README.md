# SEMD ML Service - Malicious URL Detection

Advanced ML microservice with continuous fine-tuning, dynamic feature engineering, and multi-stage feature selection pipeline.

## Project Structure

The project follows a modular architecture:

```
semd-ml/
├── main.py                          # Root entry point (delegates to src/cli/main.py)
├── docker-compose.yml               # Docker orchestration
├── Dockerfile                       # ML service container
├── requirements.txt                 # Python dependencies
├── README.md                        # Quick start guide
├── ML_SERVICE_GUIDE.md             # Comprehensive documentation
├── OPTIMIZATION_GUIDE.md           # Feature engineering guide
├── PROJECT_STRUCTURE.md            # This file
├── .env.example                    # Environment configuration template
│
├── dataset/                        # Training datasets
│   └── (CSV/Excel files)
│
├── models/                         # Trained model artifacts
│   └── (model_*.pkl, scaler_*.pkl, etc.)
│
├── reports/                        # Training reports
│   └── (training_report_*.json)
│
└── src/                           # Source code modules
    ├── __init__.py                # Backward compatibility exports
    │
    ├── core/                      # Core configuration and utilities
    │   ├── __init__.py
    │   ├── config.py              # Settings and feature configuration
    │   └── logger.py              # Logging setup
    │
    ├── features/                  # Feature engineering
    │   ├── __init__.py
    │   ├── features.yaml          # Feature definitions
    │   └── feature_extractor.py   # Feature extraction logic
    │
    ├── data/                      # Data loading and preprocessing
    │   ├── __init__.py
    │   └── dataset_pipeline.py    # Dataset loading, validation, balancing
    │
    ├── ml/                        # Machine learning pipeline
    │   ├── __init__.py
    │   ├── ml_pipeline.py         # ML training and evaluation
    │   ├── training_service.py    # Training orchestration
    │   └── prediction_service.py  # Prediction service
    │
    ├── infra/                     # Infrastructure clients
    │   ├── __init__.py
    │   ├── database.py            # PostgreSQL client
    │   └── redis_client.py        # Redis client
    │
    ├── tracking/                  # Experiment tracking
    │   ├── __init__.py
    │   └── mlflow_tracker.py      # MLflow integration
    │
    ├── queue/                     # Queue workers
    │   ├── __init__.py
    │   └── queue_worker.py        # Redis queue worker
    │
    └── cli/                       # Command-line interface
        ├── __init__.py
        ├── main.py                # CLI entry point
        └── cli_commands.py        # Command implementations
```

See `PROJECT_STRUCTURE.md` for detailed documentation.

## Setup

### Python Environment

```bash
python3 -m venv .venv

# Windows
venv\Scripts\activate

# Linux & MacOS
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

## CLI Usage

**All commands must be run from the `src` directory:**

```bash
cd src
```

### Train Models

```bash
# Train Decision Tree model
python3 main.py train \
  --dataset-files dataset/raw \
  --algorithms decision_tree \
  --run-name decision_tree_test \
  --output ../reports/decision_tree_results.json

# Train SVM model
python3 main.py train \
  --dataset-files dataset/raw \
  --algorithms svm \
  --run-name svm_test \
  --output ../reports/svm_results.json

# Train Random Forest model
python3 main.py train \
  --dataset-files dataset/raw \
  --algorithms random_forest \
  --run-name random_forest_test \
  --output ../reports/random_forest_results.json

# Train Xgboost model
python3 main.py train \
  --dataset-files dataset/raw \
  --algorithms xgboost \
  --run-name xgboost_test \
  --output ../reports/xgboost_results.json

# Train all algorithms
python3 main.py train \
  --dataset-files dataset/raw \
  --algorithms decision_tree random_forest xgboost svm \
  --run-name full_training
```

### Predict URLs

```bash
# Single URL
python ./main.py predict \
  --url faqs.org/people-search/rouleau-new-hampshire \
  --model-id e914569dc2a046ff93dd27fc4f506c63 \
  --output ../reports/prediction.json

# Batch from file
python main.py predict \
  --url-file urls.txt \
  --model-id run_abc123 \
  --output ../reports/predictions.json
```

### Evaluate Models

```bash
python main.py evaluate \
  --dataset-files dataset/malicious_url_test2.csv \
  --algorithms random_forest xgboost \
  --output ../reports/evaluation.json
```

### Feature Engineering Analysis

```bash
python main.py feature-engineering \
  --url "https://example.com" \
  --output ../reports/feature_analysis.json
```

## Redis worker

### Start Queue Worker

```bash
# Combined mode (training + prediction)
python main.py worker --mode combined

# Training only
python main.py worker --mode training

# Prediction only
python main.py worker --mode prediction
```

### Queue train

To submit training jobs asynchronously, push JSON messages to the Redis queue 'ml_training_queue'. Example:

```bash
redis-cli LPUSH ml_training_queue '{
  "service_conf_id": 1,
  "dataset_files": ["dataset/malicious_url_train.csv"],
  "algorithms": ["random_forest", "xgboost"],
  "run_name": "async_training_run",
  "balance_method": "smote"
}'
```

For predictions, push to 'ml_prediction_queue':

```bash
redis-cli LPUSH ml_prediction_queue '{
  "url": "https://example.com",
  "user_id": "user123",
  "model_id": "run_abc123"
}'
```

### Check Queue Status

```bash
python main.py queue-status
```

### Verify Imports

```bash
python verify_imports.py
```

## Services Setup

```bash
# Make script executable
chmod +x start_all_services.sh

# Start all Docker services
./start_all_services.sh start

# Check status
./start_all_services.sh status

# View logs
./start_all_services.sh logs mlflow
./start_all_services.sh logs backend

# Stop services
./start_all_services.sh stop

# Restart services
./start_all_services.sh restart
```

## ML Flow

### Fix permissions

```bash
sudo chown -R 1001:1001 /home/semd/.mlflow
```

## Configuration Options

### Feature Selection

```env
ENABLE_FEATURE_SELECTION=true
FEATURE_SELECTION_K=50
ENABLE_CORRELATION_FILTER=true
CORRELATION_THRESHOLD=0.95
ENABLE_VARIANCE_THRESHOLD=true
VARIANCE_THRESHOLD=0.01
ENABLE_MUTUAL_INFORMATION=true
MUTUAL_INFO_THRESHOLD=0.01
```

### Feature Importance

```env
ENABLE_FEATURE_IMPORTANCE=true
ENABLE_CLASS_WEIGHTING=true
CLASS_WEIGHT_MODE=soft
```

## Docker Deployment

```bash
docker-compose up -d
```

This starts:
- ML Service worker
- Redis
- PostgreSQL
- MLflow server

## Dataset Resources

- https://urlhaus.abuse.ch/browse/
- https://www.phishtank.com/developer_info.php
- https://huggingface.co/datasets/JorgeGMM/malicious_urls
- https://huggingface.co/datasets/EustassKidman/malicious-url/viewer/default/train