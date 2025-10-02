#!/bin/bash
set -e

# Function to wait for services
wait_for_service() {
    host="$1"
    port="$2"
    echo "Waiting for $host:$port..."
    while ! nc -z "$host" "$port"; do
        sleep 1
    done
    echo "$host:$port is available"
}

# Run based on command
case "$1" in
    api)
        echo "Starting API server..."
        exec uvicorn api.server:app --host 0.0.0.0 --port 8000
        ;;
    
    worker)
        echo "Starting background worker..."
        exec python scripts/worker.py
        ;;
    
    script)
        shift
        echo "Running script: $@"
        exec python "$@"
        ;;
    
    bash)
        exec /bin/bash
        ;;
    
    *)
        echo "Running custom command: $@"
        exec "$@"
        ;;
esac