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
# Source de vérité du tag [D-M.3] : arg CLI > .env > erreur.
if [ -f .env ]; then
  set -a; source .env; set +a
fi
TAG="${1:-${TRAINER_IMAGE_TAG:?TAG manquant : passer en arg ou définir TRAINER_IMAGE_TAG dans .env}}"
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