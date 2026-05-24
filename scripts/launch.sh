#!/usr/bin/env bash
set -euo pipefail

# Usage: ./launch.sh <path-to-docker-compose-dir>
# Example: ./launch.sh models/phi-3/vllm/compose/

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-docker-compose-dir>"
  echo "Example: $0 models/phi-3/vllm/compose/"
  exit 1
fi

COMPOSE_DIR="$1"

if [[ ! -d "$COMPOSE_DIR" ]]; then
  echo "Error: Directory not found: $COMPOSE_DIR"
  exit 1
fi

if [[ ! -f "$COMPOSE_DIR/docker-compose.yml" ]]; then
  echo "Error: docker-compose.yml not found in $COMPOSE_DIR"
  exit 1
fi

echo "Starting service from $COMPOSE_DIR..."
cd "$COMPOSE_DIR"
docker compose up -d
echo "Service started. Check logs with: docker compose logs -f"
echo "API should be available at http://localhost:8000/v1/chat/completions"