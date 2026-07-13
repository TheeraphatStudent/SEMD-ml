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
SRC_DIR := $(SCRIPT_DIR)/src
BACKEND_DIR := $(abspath $(SCRIPT_DIR)/../semd-backend)
FALLBACK_BACKEND_DIR := $(abspath $(SCRIPT_DIR)/../semd-shared-network)
ML_ENTRY := $(UV_RUN) python main.py

# CLI argument variables — override on the command line, e.g.:
#   make train ALGORITHMS="svm xgboost" RUN_NAME=my_run
DATASET_FILES ?= dataset/raw
ALGORITHMS    ?=
BALANCE       ?=
RUN_NAME      ?=
OUTPUT        ?=
MODEL_ID      ?=
URL           ?=
URLS          ?=
URL_FILE      ?=
CSV           ?=
STORE_PATH    ?=
RAW_PATH      ?=
CONFIG        ?=
MODE          ?= combined
ARGS          ?=

.PHONY: help venv run verify-imports mlflow-permissions start stop status restart logs \
        cli ml-help train train-obo predict predict-test evaluate feature-engineering \
        worker queue-status data-migrate data-migrate-feature

help:
	@echo "SEMD ML Makefile"
	@echo ""
	@echo "Infra targets:"
	@echo "  make venv                         Create .venv and install requirements with uv"
	@echo "  make run ARGS='verify_imports.py' Run an arbitrary python file through uv run"
	@echo "  make verify-imports               Run verify_imports.py through uv run"
	@echo "  make mlflow-permissions           Create MLflow dirs and set permissions"
	@echo "  make start                        Start database, backend, and MLflow"
	@echo "  make stop                         Stop database, backend, and MLflow"
	@echo "  make status                       Show container and port status"
	@echo "  make restart                      Restart all services"
	@echo "  make logs LOG_SERVICE=mlflow      Follow logs: mlflow, backend, database"
	@echo ""
	@echo "ML CLI targets (wrap 'uv run main.py <command>' from src/):"
	@echo "  make train DATASET_FILES=... ALGORITHMS=... RUN_NAME=... [OUTPUT=...]"
	@echo "  make train-obo [STORE_PATH=...] [ALGORITHMS=...] [RUN_NAME=...]"
	@echo "  make predict URL=... [MODEL_ID=...] [OUTPUT=...]"
	@echo "  make predict-test URL=... | CSV=... [MODEL_ID=...]"
	@echo "  make evaluate DATASET_FILES=... [ALGORITHMS=...]"
	@echo "  make feature-engineering URL=..."
	@echo "  make worker [MODE=combined|training|prediction]"
	@echo "  make queue-status                 Show Redis queue status"
	@echo "  make data-migrate                 Extract datasets from dataset/store archives"
	@echo "  make data-migrate-feature         Migrate feature reference CSVs"
	@echo "  make cli ARGS='predict --url ...' Passthrough for any main.py subcommand"
	@echo "  make ml-help [ARGS=train]         Show main.py --help (or a subcommand's --help)"

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

	source .venv/bin/activate

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
	BACKEND_COMPOSE_DIR="$$BACKEND_DIR"
	if [ -f "$$BACKEND_DIR/docker/compose.yaml" ]; then BACKEND_COMPOSE_DIR="$$BACKEND_DIR/docker"; fi
	echo ""
	echo "[1/3] Backend Database (PostgreSQL, Redis)..."
	echo "----------------------------------------"
	if nc -z localhost 5432 2>/dev/null && nc -z localhost 6379 2>/dev/null; then
		echo "[OK] PostgreSQL and Redis already running, connecting to existing instance"
	else
		if [ ! -f "$$BACKEND_DIR/database/docker-compose.database.yaml" ]; then
			echo "[ERR] Backend database compose file not found: $$BACKEND_DIR/database/docker-compose.database.yaml"
			exit 1
		fi
		cd "$$BACKEND_DIR/database"
		$$COMPOSE_CMD -f docker-compose.database.yaml up -d
		wait_for_service "PostgreSQL" localhost 5432 30
		wait_for_service "Redis" localhost 6379 30
	fi
	echo ""
	echo "[2/3] Backend Services..."
	echo "----------------------------------------"
	if nc -z localhost 8000 2>/dev/null; then
		echo "[OK] Backend API already running, connecting to existing instance"
	else
		if [ ! -f "$$BACKEND_COMPOSE_DIR/compose.yaml" ]; then
			echo "[ERR] Backend compose file not found: $$BACKEND_COMPOSE_DIR/compose.yaml"
			exit 1
		fi
		cd "$$BACKEND_COMPOSE_DIR"
		$$COMPOSE_CMD -f compose.yaml up -d
		wait_for_service "Backend API" localhost 8000 30
	fi
	echo ""
	echo "[3/3] MLflow Server..."
	echo "----------------------------------------"
	if nc -z localhost 5000 2>/dev/null; then
		echo "[OK] MLflow already running, connecting to existing instance"
	else
		if [ ! -f "$$ML_COMPOSE_DIR/docker-compose.yml" ]; then
			echo "[ERR] MLflow compose file not found in $$SCRIPT_DIR or $$SCRIPT_DIR/docker"
			exit 1
		fi
		cd "$$ML_COMPOSE_DIR"
		$$COMPOSE_CMD up -d mlflow
		wait_for_service "MLflow" localhost 5000 60
	fi
	$${MAKE:-make} --no-print-directory -C "$(SCRIPT_DIR)" status
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
	BACKEND_COMPOSE_DIR="$$BACKEND_DIR"
	if [ -f "$$BACKEND_DIR/docker/compose.yaml" ]; then BACKEND_COMPOSE_DIR="$$BACKEND_DIR/docker"; fi
	echo "Stopping all services..."
	echo ""
	echo "Stopping MLflow..."
	cd "$$ML_COMPOSE_DIR"
	$$COMPOSE_CMD down
	echo "Stopping Backend services..."
	cd "$$BACKEND_COMPOSE_DIR"
	$$COMPOSE_CMD -f compose.yaml down
	echo "Stopping Backend database..."
	cd "$$BACKEND_DIR/database"
	$$COMPOSE_CMD -f docker-compose.database.yaml down
	echo ""
	echo "[OK] All services stopped"
	$${MAKE:-make} --no-print-directory -C "$(SCRIPT_DIR)" status

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
	BACKEND_COMPOSE_DIR="$$BACKEND_DIR"
	if [ -f "$$BACKEND_DIR/docker/compose.yaml" ]; then BACKEND_COMPOSE_DIR="$$BACKEND_DIR/docker"; fi
	case "$(LOG_SERVICE)" in
		mlflow)
			cd "$$ML_COMPOSE_DIR"
			$$COMPOSE_CMD logs -f mlflow
			;;
		backend)
			cd "$$BACKEND_COMPOSE_DIR"
			$$COMPOSE_CMD -f compose.yaml logs -f
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

cli:
	if [ -z "$${ARGS:-}" ]; then \
		echo "Usage: make cli ARGS='predict --url https://example.com'"; \
		exit 1; \
	fi
	cd $(SRC_DIR) && $(ML_ENTRY) $$ARGS

ml-help:
	cd $(SRC_DIR) && $(ML_ENTRY) $(if $(ARGS),$(ARGS)) --help

train:
	cd $(SRC_DIR) && $(ML_ENTRY) train \
		--dataset-files $(DATASET_FILES) \
		$(if $(ALGORITHMS),--algorithms $(ALGORITHMS)) \
		$(if $(BALANCE),--balance $(BALANCE)) \
		$(if $(RUN_NAME),--run-name $(RUN_NAME)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

train-obo:
	cd $(SRC_DIR) && $(ML_ENTRY) train-obo \
		$(if $(STORE_PATH),--store-path $(STORE_PATH)) \
		$(if $(ALGORITHMS),--algorithms $(ALGORITHMS)) \
		$(if $(BALANCE),--balance $(BALANCE)) \
		$(if $(RUN_NAME),--run-name $(RUN_NAME)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

predict:
	if [ -z "$(URL)$(URLS)$(URL_FILE)" ]; then \
		echo "Usage: make predict URL='https://example.com' [MODEL_ID=run_abc123]"; \
		exit 1; \
	fi
	cd $(SRC_DIR) && $(ML_ENTRY) predict \
		$(if $(URL),--url "$(URL)") \
		$(if $(URLS),--urls $(URLS)) \
		$(if $(URL_FILE),--url-file $(URL_FILE)) \
		$(if $(MODEL_ID),--model-id $(MODEL_ID)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

predict-test:
	if [ -z "$(URL)$(URLS)$(CSV)" ]; then \
		echo "Usage: make predict-test URL='https://example.com' | CSV=urls.csv"; \
		exit 1; \
	fi
	cd $(SRC_DIR) && $(ML_ENTRY) predict-test \
		$(if $(URL),--url "$(URL)") \
		$(if $(URLS),--urls $(URLS)) \
		$(if $(CSV),--csv $(CSV)) \
		$(if $(MODEL_ID),--model-id $(MODEL_ID)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

evaluate:
	cd $(SRC_DIR) && $(ML_ENTRY) evaluate \
		--dataset-files $(DATASET_FILES) \
		$(if $(ALGORITHMS),--algorithms $(ALGORITHMS)) \
		$(if $(BALANCE),--balance $(BALANCE)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

feature-engineering:
	cd $(SRC_DIR) && $(ML_ENTRY) feature-engineering \
		$(if $(URL),--url "$(URL)") \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

worker:
	cd $(SRC_DIR) && $(ML_ENTRY) worker --mode $(MODE) $(ARGS)

queue-status:
	cd $(SRC_DIR) && $(ML_ENTRY) queue-status

data-migrate:
	cd $(SRC_DIR) && $(ML_ENTRY) data-migrate \
		$(if $(STORE_PATH),--store-path $(STORE_PATH)) \
		$(if $(RAW_PATH),--raw-path $(RAW_PATH)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)

data-migrate-feature:
	cd $(SRC_DIR) && $(ML_ENTRY) data-migrate-feature \
		$(if $(STORE_PATH),--store-path $(STORE_PATH)) \
		$(if $(RAW_PATH),--raw-path $(RAW_PATH)) \
		$(if $(CONFIG),--config $(CONFIG)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(ARGS)
