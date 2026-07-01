ifeq ($(OS),Windows_NT)
SHELL := C:/Program Files/Git/bin/bash.exe
else
SHELL := /bin/bash
endif
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

.DEFAULT_GOAL := help

ML_USER ?= semd
ML_GROUP ?= semd
LOG_SERVICE ?= mlflow
UV ?= uv
UV_RUN := $(UV) run
UV_PIP := $(UV) pip

SCRIPT_DIR := $(CURDIR)
BACKEND_DIR := $(abspath $(SCRIPT_DIR)/../semd-backend)
FALLBACK_BACKEND_DIR := $(abspath $(SCRIPT_DIR)/../semd-shared-network)

.PHONY: help venv run verify-imports mlflow-permissions start stop status restart logs

help:
	@echo "SEMD ML Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make venv                         Create .venv and install requirements with uv"
	@echo "  make run CMD='python ...'         Run an arbitrary command through uv run"
	@echo "  make verify-imports               Run verify_imports.py through uv run"
	@echo "  make mlflow-permissions           Create MLflow dirs and set permissions"
	@echo "  make start                        Start database, backend, and MLflow"
	@echo "  make stop                         Stop database, backend, and MLflow"
	@echo "  make status                       Show container and port status"
	@echo "  make restart                      Restart all services"
	@echo "  make logs LOG_SERVICE=mlflow      Follow logs: mlflow, backend, database"

venv:
	if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "uv is not installed or not on PATH. Install uv first: https://docs.astral.sh/uv/"; \
		exit 1; \
	fi
	if [ -n "$${VIRTUAL_ENV:-}" ]; then \
		deactivate 2>/dev/null || true; \
	fi
	if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment with uv..."; \
		$(UV) venv .venv; \
	fi
	if [ -f ".venv/bin/activate" ]; then \
		VENV_BIN=".venv/bin"; \
		VENV_ACTIVATE=".venv/bin/activate"; \
	elif [ -f ".venv/Scripts/activate" ]; then \
		VENV_BIN=".venv/Scripts"; \
		VENV_ACTIVATE=".venv/Scripts/activate"; \
	else \
		echo "Failed to create virtual environment. Ensure uv is working."; \
		exit 1; \
	fi
	if [ -d ".venv/bin" ] && [ ! -x ".venv/bin/activate" ]; then \
		chmod -R +x .venv/bin/; \
	fi
	source "$$VENV_ACTIVATE"
	$(UV_PIP) install -r requirements.txt
	$(UV_PIP) list
	echo "Environment is ready at .venv"

run:
	if [ -z "$${ARGS:-}" ]; then \
		echo "Usage: make run ARGS='verify_imports.py'"; \
		exit 1; \
	fi
	$(UV_RUN) python $$ARGS

verify-imports:
	$(UV_RUN) python verify_imports.py

mlflow-permissions:
	echo "Setting up MLflow directories and permissions..."
	mkdir -p ./mlflow_data/artifacts/models
	mkdir -p ./mlflow_data/artifacts/reports
	mkdir -p ./models
	mkdir -p ./reports
	sudo chown -R $(ML_USER):$(ML_GROUP) ./models ./reports ./mlflow_data
	chmod -R 775 ./mlflow_data
	chmod -R 775 ./models
	chmod -R 775 ./reports
	echo "MLflow directories and permissions set up successfully."
	echo "Directory structure:"
	echo "- ./mlflow_data (MLflow database and artifacts)"
	echo "- ./models (Model artifacts)"
	echo "- ./reports (Training reports)"
	if command -v docker >/dev/null 2>&1; then \
		echo ""; \
		echo "Docker is available. You can now run: make start"; \
	elif command -v podman >/dev/null 2>&1; then \
		echo ""; \
		echo "Podman is available. You can now run: make start"; \
	else \
		echo ""; \
		echo "Docker or Podman not found. Please install Docker or Podman to run MLflow."; \
	fi

start:
	SCRIPT_DIR="$(SCRIPT_DIR)"
	BACKEND_DIR="$(BACKEND_DIR)"
	if [ ! -d "$$BACKEND_DIR" ]; then BACKEND_DIR="$(FALLBACK_BACKEND_DIR)"; fi
	ML_COMPOSE_DIR="$$SCRIPT_DIR"
	if [ -f "$$SCRIPT_DIR/docker/docker-compose.yml" ]; then ML_COMPOSE_DIR="$$SCRIPT_DIR/docker"; fi
	CONTAINER_CMD=""
	COMPOSE_CMD=""
	detect_container_runtime() {
		if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
			CONTAINER_CMD="docker"
			if command -v docker-compose >/dev/null 2>&1; then
				COMPOSE_CMD="docker-compose"
			elif docker compose version >/dev/null 2>&1; then
				COMPOSE_CMD="docker compose"
			fi
			echo "[OK] Using Docker"
			return 0
		fi
		if command -v podman >/dev/null 2>&1; then
			CONTAINER_CMD="podman"
			if command -v podman compose >/dev/null 2>&1; then
				COMPOSE_CMD="podman compose"
			else
				echo "[ERR] Podman found but podman compose is not installed"
				return 1
			fi
			echo "[OK] Using Podman (Docker not available)"
			return 0
		fi
		echo "[ERR] Neither Docker nor Podman is installed"
		echo "      Install Docker: https://docs.docker.com/get-docker/"
		echo "      Or install Podman: https://podman.io/getting-started/installation"
		return 1
	}
	check_compose() {
		if [ -z "$$COMPOSE_CMD" ]; then
			echo "[ERR] Compose tool not available"
			return 1
		fi
		echo "[OK] $$COMPOSE_CMD is ready"
	}
	wait_for_service() {
		local service_name="$$1"
		local host="$$2"
		local port="$$3"
		local max_wait="$${4:-60}"
		echo "Waiting for $$service_name to be ready..."
		for i in $$(seq 1 "$$max_wait"); do
			if nc -z "$$host" "$$port" 2>/dev/null; then
				echo "[OK] $$service_name is ready on $$host:$$port"
				return 0
			fi
			sleep 1
			if [ $$((i % 10)) -eq 0 ]; then
				echo "  Still waiting... ($$i/$$max_wait seconds)"
			fi
		done
		echo "[ERR] $$service_name failed to start within $$max_wait seconds"
		return 1
	}
	echo "=========================================="
	echo "SEMD ML Service - Starting All Services"
	echo "=========================================="
	echo ""
	echo "Project structure:"
	echo "  Backend: $$BACKEND_DIR"
	echo "  ML Service: $$SCRIPT_DIR"
	echo ""
	detect_container_runtime
	check_compose
	echo ""
	echo "[1/3] Starting Backend Database (PostgreSQL, Redis)..."
	echo "----------------------------------------"
	if [ ! -f "$$BACKEND_DIR/database/docker-compose.database.yaml" ]; then
		echo "[ERR] Backend database compose file not found: $$BACKEND_DIR/database/docker-compose.database.yaml"
		exit 1
	fi
	cd "$$BACKEND_DIR/database"
	$$COMPOSE_CMD -f docker-compose.database.yaml up -d
	wait_for_service "PostgreSQL" localhost 5432 30
	wait_for_service "Redis" localhost 6379 30
	echo ""
	echo "[2/3] Starting Backend Services..."
	echo "----------------------------------------"
	if [ ! -f "$$BACKEND_DIR/compose.yaml" ]; then
		echo "[ERR] Backend compose file not found: $$BACKEND_DIR/compose.yaml"
		exit 1
	fi
	cd "$$BACKEND_DIR"
	$$COMPOSE_CMD -f compose.yaml up -d
	wait_for_service "Backend API" localhost 8000 30
	echo ""
	echo "[3/3] Starting MLflow Server..."
	echo "----------------------------------------"
	if [ ! -f "$$ML_COMPOSE_DIR/docker-compose.yml" ]; then
		echo "[ERR] MLflow compose file not found in $$SCRIPT_DIR or $$SCRIPT_DIR/docker"
		exit 1
	fi
	cd "$$ML_COMPOSE_DIR"
	$$COMPOSE_CMD up -d mlflow
	wait_for_service "MLflow" localhost 5000 60
	$${MAKE:-make} --no-print-directory status
	echo ""
	echo "[OK] All services started."
	echo ""
	echo "Service URLs:"
	echo "  - Backend API: http://localhost:8000"
	echo "  - MLflow UI: http://localhost:5000"
	echo "  - PostgreSQL: localhost:5432"
	echo "  - Redis: localhost:6379"
	echo ""
	echo "Next steps:"
	echo "  cd src"
	echo "  uv run python verify_imports.py"
	echo "  uv run python main.py train --dataset-files dataset/malicious_urls_train1.csv --algorithms svm"

stop:
	SCRIPT_DIR="$(SCRIPT_DIR)"
	BACKEND_DIR="$(BACKEND_DIR)"
	if [ ! -d "$$BACKEND_DIR" ]; then BACKEND_DIR="$(FALLBACK_BACKEND_DIR)"; fi
	ML_COMPOSE_DIR="$$SCRIPT_DIR"
	if [ -f "$$SCRIPT_DIR/docker/docker-compose.yml" ]; then ML_COMPOSE_DIR="$$SCRIPT_DIR/docker"; fi
	COMPOSE_CMD=""
	if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
		if command -v docker-compose >/dev/null 2>&1; then COMPOSE_CMD="docker-compose"; else COMPOSE_CMD="docker compose"; fi
	elif command -v podman compose >/dev/null 2>&1; then
		COMPOSE_CMD="podman compose"
	else
		echo "[ERR] No compose tool available"
		exit 1
	fi
	echo "Stopping all services..."
	echo ""
	echo "Stopping MLflow..."
	cd "$$ML_COMPOSE_DIR"
	$$COMPOSE_CMD down
	echo "Stopping Backend services..."
	cd "$$BACKEND_DIR"
	$$COMPOSE_CMD -f compose.yaml down
	echo "Stopping Backend database..."
	cd "$$BACKEND_DIR/database"
	$$COMPOSE_CMD -f docker-compose.database.yaml down
	echo ""
	echo "[OK] All services stopped"
	$${MAKE:-make} --no-print-directory status

status:
	SCRIPT_DIR="$(SCRIPT_DIR)"
	BACKEND_DIR="$(BACKEND_DIR)"
	if [ ! -d "$$BACKEND_DIR" ]; then BACKEND_DIR="$(FALLBACK_BACKEND_DIR)"; fi
	CONTAINER_CMD=""
	if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
		CONTAINER_CMD="docker"
	elif command -v podman >/dev/null 2>&1; then
		CONTAINER_CMD="podman"
	else
		echo "[ERR] Neither Docker nor Podman is available"
		exit 1
	fi
	echo ""
	echo "=========================================="
	echo "Service Status"
	echo "=========================================="
	echo ""
	echo "Backend Database:"
	$$CONTAINER_CMD ps --filter "name=postgres" --format "  {{.Names}}: {{.Status}}" || true
	$$CONTAINER_CMD ps --filter "name=redis" --format "  {{.Names}}: {{.Status}}" || true
	echo ""
	echo "Backend Services:"
	$$CONTAINER_CMD ps --filter "name=backend" --format "  {{.Names}}: {{.Status}}" || true
	$$CONTAINER_CMD ps --filter "name=mlflow" --format "  {{.Names}}: {{.Status}}" || true
	echo ""
	echo "MLflow:"
	$$CONTAINER_CMD ps --filter "name=semd-mlflow" --format "  {{.Names}}: {{.Status}}" || true
	echo ""
	echo "Port Status:"
	nc -z localhost 5432 2>/dev/null && echo "  [OK] PostgreSQL: localhost:5432" || echo "  [--] PostgreSQL: localhost:5432"
	nc -z localhost 6379 2>/dev/null && echo "  [OK] Redis: localhost:6379" || echo "  [--] Redis: localhost:6379"
	nc -z localhost 8000 2>/dev/null && echo "  [OK] Backend API: localhost:8000" || echo "  [--] Backend API: localhost:8000"
	nc -z localhost 5000 2>/dev/null && echo "  [OK] MLflow: localhost:5000" || echo "  [--] MLflow: localhost:5000"
	echo "=========================================="

restart:
	$${MAKE:-make} --no-print-directory stop
	sleep 3
	$${MAKE:-make} --no-print-directory start

logs:
	SCRIPT_DIR="$(SCRIPT_DIR)"
	BACKEND_DIR="$(BACKEND_DIR)"
	if [ ! -d "$$BACKEND_DIR" ]; then BACKEND_DIR="$(FALLBACK_BACKEND_DIR)"; fi
	ML_COMPOSE_DIR="$$SCRIPT_DIR"
	if [ -f "$$SCRIPT_DIR/docker/docker-compose.yml" ]; then ML_COMPOSE_DIR="$$SCRIPT_DIR/docker"; fi
	COMPOSE_CMD=""
	if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
		if command -v docker-compose >/dev/null 2>&1; then COMPOSE_CMD="docker-compose"; else COMPOSE_CMD="docker compose"; fi
	elif command -v podman compose >/dev/null 2>&1; then
		COMPOSE_CMD="podman compose"
	else
		echo "[ERR] No compose tool available"
		exit 1
	fi
	case "$(LOG_SERVICE)" in
		mlflow)
			cd "$$ML_COMPOSE_DIR"
			$$COMPOSE_CMD logs -f mlflow
			;;
		backend)
			cd "$$BACKEND_DIR"
			$$COMPOSE_CMD logs -f
			;;
		database)
			cd "$$BACKEND_DIR/database"
			$$COMPOSE_CMD logs -f
			;;
		*)
			echo "Available logs: mlflow, backend, database"
			exit 1
			;;
	esac
