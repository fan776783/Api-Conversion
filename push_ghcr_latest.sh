#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/fan776783/api-conversion:latest"
GHCR_USER="fan776783"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if [[ -n "${GHCR_PAT:-}" ]]; then
  printf '%s' "$GHCR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
else
  echo "GHCR_PAT is not set, skipping docker login."
  echo "If this is your first push, run: docker login ghcr.io -u $GHCR_USER"
fi

echo "Building image: $IMAGE"
docker build -t "$IMAGE" -f Dockerfile .

echo "Pushing image: $IMAGE"
docker push "$IMAGE"

echo "Done: $IMAGE"
