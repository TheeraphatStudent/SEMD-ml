#!/bin/bash

echo "Setting up MLflow directories and permissions..."

mkdir -p ./mlflow_data/artifacts/models
mkdir -p ./mlflow_data/artifacts/reports
mkdir -p ./models
mkdir -p ./reports

sudo chown -R semd:semd ./models ./reports ./mlflow_data

chmod -R 775 ./mlflow_data
chmod -R 775 ./models
chmod -R 775 ./reports

echo "MLflow directories and permissions set up successfully!"
echo "Directory structure:"
echo "- ./mlflow_data (MLflow database and artifacts)"
echo "- ./models (Model artifacts)"
echo "- ./reports (Training reports)"

if command -v docker &> /dev/null; then
    echo ""
    echo "Docker is available. You can now run:"
    echo "docker-compose up -d mlflow"
elif command -v podman &> /dev/null; then
    echo ""
    echo "Podman is available. You can now run:"
    echo "podman-compose up -d mlflow"
else
    echo ""
    echo "Docker or Podman not found. Please install Docker or Podman to run MLflow."
fi
