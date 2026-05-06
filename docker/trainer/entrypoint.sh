#!/bin/bash
set -euo pipefail

cd /workspace

echo "==================================="
echo "MLOps Rakuten Trainer — démarrage"
echo "==================================="
echo "Hostname : $(hostname)"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'aucun')"
echo "Python : $(python --version)"
echo "==================================="

# === 1. Configuration DVC pour R2 ===
if [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
    echo "ERROR: R2_ACCESS_KEY_ID et R2_SECRET_ACCESS_KEY doivent être définis"
    exit 1
fi

# === 2. Init Git minimal pour DVC (pas de SCM dans l'image) ===
if [ ! -d ".git" ]; then
    echo "[Entrypoint] Init Git minimal pour DVC..."
    git init -q
    git config user.email "trainer@runpod"
    git config user.name "trainer"
fi

dvc remote modify --local r2 access_key_id "$R2_ACCESS_KEY_ID"
dvc remote modify --local r2 secret_access_key "$R2_SECRET_ACCESS_KEY"

# === 3. Cache DVC sur le volume persistant si monté ===
if [ -d "/workspace/cache" ]; then
    mkdir -p /workspace/cache/dvc-cache
    dvc cache dir /workspace/cache/dvc-cache
    echo "[Entrypoint] Cache DVC sur volume persistant : /workspace/cache/dvc-cache"
else
    echo "[Entrypoint] Pas de volume persistant, cache DVC éphémère"
fi

# === 4. DVC pull des données ===
echo "[Entrypoint] DVC pull..."
if [ -n "${DVC_PULL_TARGETS:-}" ]; then
    # shellcheck disable=SC2086
    dvc pull $DVC_PULL_TARGETS -v
else
    dvc pull -v
fi

# === 5. Exécution de la commande ===
echo "[Entrypoint] Exécution : $*"
"$@"
EXIT_CODE=$?

# === 6. DVC push si succès et DVC_AUTO_PUSH=true ===
if [ "$EXIT_CODE" -eq 0 ] && [ "${DVC_AUTO_PUSH:-false}" = "true" ]; then
    echo "[Entrypoint] DVC push des nouveaux artefacts..."
    if [ -d "data/cache" ]; then
        dvc add data/cache 2>/dev/null || true
        dvc push -v || echo "[Entrypoint] WARN: dvc push a échoué"
    fi
else
    echo "[Entrypoint] Pas de push (exit=$EXIT_CODE, DVC_AUTO_PUSH=${DVC_AUTO_PUSH:-false})"
fi

exit $EXIT_CODE