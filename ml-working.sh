#!/bin/bash

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi

if ! dpkg -l | grep -q python3-venv; then
    echo "python3-venv not found. Installing python3.12-venv..."
    if ! sudo apt update && sudo apt install -y python3.12-venv; then
        echo "Failed to install python3.12-venv. Please install manually and rerun."
        exit 1
    fi
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "Failed to create virtual environment. Ensure python3 and venv are working."
    exit 1
fi

if [ ! -x ".venv/bin/activate" ]; then
    chmod -R +x .venv/bin/
fi

source .venv/bin/activate

./.venv/bin/pip install -r requirements.txt

./.venv/bin/pip list

echo "Activated environment at \"ml\""