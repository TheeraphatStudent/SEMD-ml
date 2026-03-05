#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")/semd-backend"

if [ ! -d "$BACKEND_DIR" ]; then
    BACKEND_DIR="$(dirname "$SCRIPT_DIR")/semd-shared-network"
fi

CONTAINER_CMD=""
COMPOSE_CMD=""

echo "=========================================="
echo "SEMD ML Service - Starting All Services"
echo "=========================================="
echo ""
echo "Project structure:"
echo "  Backend: $BACKEND_DIR"
echo "  ML Service: $SCRIPT_DIR"
echo ""

detect_container_runtime() {
    if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
        CONTAINER_CMD="docker"
        if command -v docker-compose &> /dev/null; then
            COMPOSE_CMD="docker-compose"
        elif docker compose version &> /dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
        fi
        echo "✓ Using Docker"
        return 0
    fi
    
    if command -v podman &> /dev/null; then
        CONTAINER_CMD="podman"
        if command -v podman-compose &> /dev/null; then
            COMPOSE_CMD="podman-compose"
        else
            echo "❌ Podman found but podman-compose not installed"
            return 1
        fi
        echo "✓ Using Podman (Docker not available)"
        return 0
    fi
    
    echo "❌ Neither Docker nor Podman is installed"
    echo "   Install Docker: https://docs.docker.com/get-docker/"
    echo "   Or install Podman: https://podman.io/getting-started/installation"
    return 1
}

check_docker() {
    detect_container_runtime
    return $?
}

check_docker_compose() {
    if [ -z "$COMPOSE_CMD" ]; then
        echo "❌ Compose tool not available"
        return 1
    fi
    echo "✓ $COMPOSE_CMD is ready"
    return 0
}

wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_wait=${4:-60}
    
    echo "Waiting for $service_name to be ready..."
    for i in $(seq 1 $max_wait); do
        if nc -z $host $port 2>/dev/null; then
            echo "✓ $service_name is ready on $host:$port"
            return 0
        fi
        sleep 1
        if [ $((i % 10)) -eq 0 ]; then
            echo "  Still waiting... ($i/$max_wait seconds)"
        fi
    done
    
    echo "❌ $service_name failed to start within $max_wait seconds"
    return 1
}

start_backend_database() {
    echo ""
    echo "[1/3] Starting Backend Database (PostgreSQL, Redis)..."
    echo "----------------------------------------"
    
    if [ ! -f "$BACKEND_DIR/database/docker-compose.database.yaml" ]; then
        echo "❌ Backend database compose file not found: $BACKEND_DIR/database/docker-compose.database.yaml"
        return 1
    fi
    
    cd "$BACKEND_DIR/database"
    
    echo "Starting containers..."
    $COMPOSE_CMD -f docker-compose.database.yaml up -d
    
    if [ $? -eq 0 ]; then
        echo "✓ Backend database containers started"
        
        wait_for_service "PostgreSQL" localhost 5432 30
        wait_for_service "Redis" localhost 6379 30
        
        return 0
    else
        echo "❌ Failed to start backend database"
        return 1
    fi
}

start_backend_services() {
    echo ""
    echo "[2/3] Starting Backend Services..."
    echo "----------------------------------------"
    
    if [ ! -f "$BACKEND_DIR/compose.yaml" ]; then
        echo "❌ Backend compose file not found: $BACKEND_DIR/compose.yaml"
        return 1
    fi
    
    cd "$BACKEND_DIR"
    
    echo "Starting containers..."
    $COMPOSE_CMD -f compose.yaml up -d
    
    if [ $? -eq 0 ]; then
        echo "✓ Backend service containers started"
        
        wait_for_service "Backend API" localhost 8000 30
        
        return 0
    else
        echo "❌ Failed to start backend services"
        return 1
    fi
}

start_mlflow() {
    echo ""
    echo "[3/3] Starting MLflow Server..."
    echo "----------------------------------------"
    
    if [ ! -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        echo "❌ MLflow compose file not found: $SCRIPT_DIR/docker-compose.yml"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    
    echo "Starting MLflow container..."
    $COMPOSE_CMD up -d mlflow
    
    if [ $? -eq 0 ]; then
        echo "✓ MLflow container started"
        
        wait_for_service "MLflow" localhost 5000 60
        
        return 0
    else
        echo "❌ Failed to start MLflow"
        return 1
    fi
}

show_status() {
    echo ""
    echo "=========================================="
    echo "Service Status"
    echo "=========================================="
    
    echo ""
    echo "Backend Database:"
    $CONTAINER_CMD ps --filter "name=postgres" --format "  {{.Names}}: {{.Status}}"
    $CONTAINER_CMD ps --filter "name=redis" --format "  {{.Names}}: {{.Status}}"
    
    echo ""
    echo "Backend Services:"
    $CONTAINER_CMD ps --filter "name=backend" --format "  {{.Names}}: {{.Status}}"
    $CONTAINER_CMD ps --filter "name=mlflow" --format "  {{.Names}}: {{.Status}}"
    
    echo ""
    echo "MLflow:"
    $CONTAINER_CMD ps --filter "name=semd-mlflow" --format "  {{.Names}}: {{.Status}}"
    
    echo ""
    echo "Port Status:"
    nc -z localhost 5432 2>/dev/null && echo "  ✓ PostgreSQL: localhost:5432" || echo "  ✗ PostgreSQL: localhost:5432"
    nc -z localhost 6379 2>/dev/null && echo "  ✓ Redis: localhost:6379" || echo "  ✗ Redis: localhost:6379"
    nc -z localhost 8000 2>/dev/null && echo "  ✓ Backend API: localhost:8000" || echo "  ✗ Backend API: localhost:8000"
    nc -z localhost 5000 2>/dev/null && echo "  ✓ MLflow: localhost:5000" || echo "  ✗ MLflow: localhost:5000"
    
    echo "=========================================="
}

stop_all_services() {
    echo ""
    echo "Stopping all services..."
    echo ""
    
    echo "Stopping MLflow..."
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD down
    
    echo "Stopping Backend services..."
    cd "$BACKEND_DIR"
    $COMPOSE_CMD -f compose.yaml down
    
    echo "Stopping Backend database..."
    cd "$BACKEND_DIR/database"
    $COMPOSE_CMD -f docker-compose.database.yaml down
    
    echo ""
    echo "✓ All services stopped"
}

show_logs() {
    local service=$1
    
    case "$service" in
        mlflow)
            cd "$SCRIPT_DIR"
            $COMPOSE_CMD logs -f mlflow
            ;;
        backend)
            cd "$BACKEND_DIR"
            $COMPOSE_CMD logs -f
            ;;
        database)
            cd "$BACKEND_DIR/database"
            $COMPOSE_CMD logs -f
            ;;
        *)
            echo "Available logs: mlflow, backend, database"
            ;;
    esac
}

case "${1:-start}" in
    start)
        check_docker || exit 1
        check_docker_compose || exit 1
        
        start_backend_database
        start_backend_services
        start_mlflow
        
        show_status
        
        echo ""
        echo "✅ All services started!"
        echo ""
        echo "Service URLs:"
        echo "  • Backend API: http://localhost:8000"
        echo "  • MLflow UI: http://localhost:5000"
        echo "  • PostgreSQL: localhost:5432"
        echo "  • Redis: localhost:6379"
        echo ""
        echo "Next steps:"
        echo "  cd src"
        echo "  python3 verify_imports.py"
        echo "  python3 main.py train --dataset-files dataset/malicious_urls_train1.csv --algorithms svm"
        ;;
    
    stop)
        stop_all_services
        show_status
        ;;
    
    status)
        show_status
        ;;
    
    restart)
        stop_all_services
        sleep 3
        check_docker || exit 1
        check_docker_compose || exit 1
        start_backend_database
        start_backend_services
        start_mlflow
        show_status
        ;;
    
    logs)
        show_logs "$2"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|status|restart|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all Docker services (database, backend, MLflow)"
        echo "  stop    - Stop all Docker services"
        echo "  status  - Check service status"
        echo "  restart - Restart all services"
        echo "  logs    - View logs (mlflow|backend|database)"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs mlflow"
        echo "  $0 status"
        exit 1
        ;;
esac
