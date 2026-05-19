#!/bin/bash
# Build + push de l'image trainer sur GHCR.
#

set -euo pipefail

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

GITHUB_USER="${GITHUB_USER:?GITHUB_USER must be set}"
GHCR_TOKEN="${GHCR_TOKEN:?GHCR_TOKEN must be set}"
IMAGE_NAME="${IMAGE_NAME:-mlops-rakuten-trainer}"
TAG="${TAG:-phase-1-l5}"
FULL_IMAGE="ghcr.io/${GITHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "==================================="
echo "Build : $FULL_IMAGE"
echo "==================================="

# Login GHCR
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

# Build (depuis la racine du projet)
docker build \
    -f docker/trainer/Dockerfile \
    -t "$FULL_IMAGE" \
    .

# Push
docker push "$FULL_IMAGE"

echo "==================================="
echo "Image disponible : $FULL_IMAGE"
echo "==================================="