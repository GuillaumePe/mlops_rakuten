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

# === 4.5. Extraction du blob images si présent ===
BLOB_PATH="/workspace/data/raw_data/images/image_train.tar.zst"
EXTRACTED_DIR="/workspace/cache/images/image_train"
SYMLINK_PATH="/workspace/data/raw_data/images/image_train"

if [ -f "$BLOB_PATH" ]; then
    # Compter le cache existant
    if [ -d "$EXTRACTED_DIR" ]; then
        N_EXISTING=$(ls "$EXTRACTED_DIR" 2>/dev/null | wc -l)
    else
        N_EXISTING=0
    fi
    echo "[$(date +%T)] Cache existant : $N_EXISTING fichiers"
    
    if [ "$N_EXISTING" -ge 84000 ]; then
        echo "[$(date +%T)] Cache complet, on réutilise"
    else
        echo "[$(date +%T)] Cache incomplet ($N_EXISTING < 84000), re-extraction..."
        
        # Extraction dans un dossier temporaire
        TEMP_DIR="/workspace/cache/images/_extract_tmp"
        rm -rf "$TEMP_DIR"
        mkdir -p "$TEMP_DIR"
        
        echo "[$(date +%T)] Décompression zstd..."
        zstd -d "$BLOB_PATH" -c | tar -xf - -C "$TEMP_DIR" --no-same-owner --no-same-permissions
        echo "[$(date +%T)] Décompression terminée"
        
        N_EXTRACTED=$(ls "$TEMP_DIR/image_train" 2>/dev/null | wc -l)
        echo "[$(date +%T)] $N_EXTRACTED fichiers extraits"
        
        if [ "$N_EXTRACTED" -lt 84000 ]; then
            echo "[ERROR] Extraction incomplète ($N_EXTRACTED < 84000), abort"
            exit 1
        fi
        
        # Swap atomique : ancien cache → temp → renommage
        echo "[$(date +%T)] Swap du cache..."
        rm -rf "$EXTRACTED_DIR"
        mv "$TEMP_DIR/image_train" "$EXTRACTED_DIR"
        rm -rf "$TEMP_DIR"
        echo "[$(date +%T)] Swap terminé"
    fi
    
    # Symlink
    rm -rf "$SYMLINK_PATH" 2>/dev/null
    ln -sfn "$EXTRACTED_DIR" "$SYMLINK_PATH"
    echo "[$(date +%T)] Symlink créé : $SYMLINK_PATH → $EXTRACTED_DIR"
    echo "[$(date +%T)] Test 5 fichiers : $(ls $SYMLINK_PATH | head -5 | tr '\n' ' ')"
fi

cd /workspace
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

# Self-terminate le pod via API RunPod
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -X POST "https://api.runpod.io/graphql" \
         -H "Authorization: Bearer $RUNPOD_API_KEY" \
         -H "Content-Type: application/json" \
         --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
         > /dev/null 2>&1 || true
fi