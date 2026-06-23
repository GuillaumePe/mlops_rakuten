#!/bin/bash
set -uo pipefail

cd /workspace

# ============================================================
# Variables de contrôle (settables par le runner via pod_env)
# ============================================================
SKIP_DVC_PULL="${SKIP_DVC_PULL:-false}"
SKIP_IMAGE_EXTRACT="${SKIP_IMAGE_EXTRACT:-false}"

# ============================================================
# Setup logging vers fichier sur volume persistant
# (toute la sortie stdout+stderr est aussi écrite dans LOG_FILE)
# ============================================================
LOG_DIR="/workspace/cache/logs"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pod_${RUNPOD_POD_ID:-unknown}_${LOG_TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[entrypoint] Logging vers : $LOG_FILE"

# ============================================================
# Cleanup unifié — appelé sur TOUT exit du script.
# C'est ce qui garantit que le pod se self-terminate même
# en cas d'échec (sinon RunPod redémarre en boucle).
# ============================================================
TAILSCALED_PID=""

cleanup_and_terminate() {
    EXIT_CODE=$?
    echo "[cleanup] === Cleanup déclenché, exit_code=$EXIT_CODE ==="

    # 1) Upload du log vers R2 (best effort, ne bloque jamais)
    if [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_ENDPOINT_URL:-}" ] && [ -f "$LOG_FILE" ]; then
        LOG_REMOTE_NAME="pod_${RUNPOD_POD_ID:-unknown}_${LOG_TIMESTAMP}_exit${EXIT_CODE}.log"
        echo "[cleanup] Upload log vers R2 : $LOG_REMOTE_NAME"
        python /workspace/scripts/r2_logs.py upload "$LOG_FILE" "$LOG_REMOTE_NAME" 2>&1 \
            || echo "[cleanup] WARN: upload log échoué"
    else
        echo "[cleanup] Skip upload log (vars R2 manquantes ou pas de log file)"
    fi

    # 2) Logout Tailscale et kill du daemon (best effort)
    if [ -n "$TAILSCALED_PID" ]; then
        echo "[cleanup] Tailscale logout..."
        tailscale logout 2>/dev/null || true
        kill "$TAILSCALED_PID" 2>/dev/null || true
    fi

    # 3) Self-terminate du pod RunPod (priorité absolue)
    if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "[cleanup] Self-terminate du pod $RUNPOD_POD_ID..."
        curl -s -X POST "https://api.runpod.io/graphql" \
             -H "Authorization: Bearer $RUNPOD_API_KEY" \
             -H "Content-Type: application/json" \
             --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" \
             > /dev/null 2>&1
        sleep 5
    else
        echo "[cleanup] Skip self-terminate (RUNPOD_API_KEY ou RUNPOD_POD_ID manquant)"
    fi
}

trap cleanup_and_terminate EXIT

# ============================================================
# Header diagnostique
# ============================================================
echo "==================================="
echo "MLOps Rakuten Trainer — démarrage"
echo "==================================="
echo "Hostname : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'aucun')"
echo "Python   : $(python --version)"
echo "==================================="

# ============================================================
# Setup Tailscale (overlay network vers MLflow local)
# ============================================================
if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
    echo "[tailscale] Démarrage tailscaled en userspace mode..."
    tailscaled \
        --tun=userspace-networking \
        --outbound-http-proxy-listen=localhost:1055 \
        --socks5-server=localhost:1055 \
        --state=/var/lib/tailscale/tailscaled.state \
        --socket=/var/run/tailscale/tailscaled.sock \
        > /tmp/tailscaled.log 2>&1 &
    TAILSCALED_PID=$!

    sleep 5

    echo "[tailscale] Authentification au tailnet (authkey préfix=${TAILSCALE_AUTHKEY:0:12}..., len=${#TAILSCALE_AUTHKEY})..."
    if ! tailscale up \
            --authkey="${TAILSCALE_AUTHKEY}" \
            --hostname="runpod-trainer-${RUNPOD_POD_ID:-unknown}" \
            --accept-routes \
            --ssh=false; then
        echo "[tailscale] ERROR: tailscale up a échoué"
        cat /tmp/tailscaled.log || true
        exit 1
    fi

    sleep 2
    echo "[tailscale] Status :"
    tailscale status || true
    echo "[tailscale] Pod IP Tailscale : $(tailscale ip -4 | head -1)"

    # ============================================================
    # Probe MLflow avec diagnostic enrichi
    # ============================================================
    if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
        echo "[tailscale] === Diagnostic connectivité avant probe ==="

        # Extraction host/port depuis l'URI
        MLFLOW_HOST=$(echo "$MLFLOW_TRACKING_URI" | sed -E 's|^https?://([^:/]+).*|\1|')
        MLFLOW_PORT=$(echo "$MLFLOW_TRACKING_URI" | sed -E 's|^https?://[^:]+:([0-9]+).*|\1|')
        echo "[diag] Target: host=$MLFLOW_HOST port=$MLFLOW_PORT"

        # 1) Tailscale ping (niveau protocole, pas TCP/HTTP)
        echo "[diag] Tailscale ping (3 paquets, 10s timeout)..."
        tailscale ping --timeout 10s -c 3 "$MLFLOW_HOST" 2>&1 | head -10 || echo "[diag] tailscale ping a échoué"

        # 2) État des peers tels que vus par tailscaled du pod
        echo "[diag] État des peers Tailscale :"
        tailscale status --peers --self=false 2>&1 | head -20 || true

        # 3) Netcheck — diagnostic réseau Tailscale général côté pod
        echo "[diag] Tailscale netcheck :"
        tailscale netcheck 2>&1 | head -25 || true

        # 4) Curl verbeux — détail du mode d'échec (Connection refused, timeout, no route...)
        echo "[diag] Curl verbose vers ${MLFLOW_TRACKING_URI}/health (timeout 15s) :"
        if curl -v --proxy http://localhost:1055 --connect-timeout 15 --max-time 15 "${MLFLOW_TRACKING_URI}/health" 2>&1; then
            echo "[tailscale] MLflow reachable via tailnet ✓"
        else
            echo "[tailscale] ERROR: MLflow injoignable, abort avant lancement job"
            exit 1
        fi
    else
        echo "[tailscale] WARN: MLFLOW_TRACKING_URI non défini, probe sautée"
    fi
else
    echo "[tailscale] TAILSCALE_AUTHKEY non défini, skip setup tailnet (mode legacy)"
fi

# ── MongoDB tunnel via SOCKS5 ──────────────────────────────────
if [ -n "${MONGO_PROXY_HOST:-}" ] && [ -n "${MONGO_URI:-}" ]; then
    MONGO_REMOTE_HOST=$(python3 -c "from urllib.parse import urlparse; print(urlparse('${MONGO_URI}').hostname)")
    MONGO_REMOTE_PORT=$(python3 -c "from urllib.parse import urlparse; print(urlparse('${MONGO_URI}').port or 27017)")
    echo "[mongo-tunnel] Démarrage tunnel localhost:27018 → ${MONGO_REMOTE_HOST}:${MONGO_REMOTE_PORT} via SOCKS5"
    python3 /workspace/mongo_tunnel.py \
        --remote-host "${MONGO_REMOTE_HOST}" \
        --remote-port "${MONGO_REMOTE_PORT}" \
        --local-port 27018 \
        --proxy-host "${MONGO_PROXY_HOST}" \
        --proxy-port "${MONGO_PROXY_PORT:-1055}" &
    sleep 1
    export MONGO_URI="mongodb://localhost:27018"
    echo "[mongo-tunnel] MONGO_URI réécrit → mongodb://localhost:27018"
fi

# ============================================================
# Configuration DVC pour R2
# ============================================================
if [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
    echo "ERROR: R2_ACCESS_KEY_ID et R2_SECRET_ACCESS_KEY doivent être définis"
    exit 1
fi

if [ ! -d ".git" ]; then
    echo "[entrypoint] Init Git minimal pour DVC..."
    git init -q
    git config user.email "trainer@runpod"
    git config user.name "trainer"
fi

dvc remote modify --local r2 access_key_id "$R2_ACCESS_KEY_ID"
dvc remote modify --local r2 secret_access_key "$R2_SECRET_ACCESS_KEY"

# Cache DVC sur le volume persistant si monté
if [ -d "/workspace/cache" ]; then
    mkdir -p /workspace/cache/dvc-cache
    dvc cache dir /workspace/cache/dvc-cache
    echo "[entrypoint] Cache DVC sur volume persistant : /workspace/cache/dvc-cache"
else
    echo "[entrypoint] Pas de volume persistant, cache DVC éphémère"
fi

# ============================================================
# DVC pull (conditionnel)
# ============================================================
if [ "$SKIP_DVC_PULL" = "true" ]; then
    echo "[entrypoint] SKIP_DVC_PULL=true, on saute le pull"
else
    echo "[entrypoint] DVC pull..."
    if [ -n "${DVC_PULL_TARGETS:-}" ]; then
        # shellcheck disable=SC2086
        dvc pull $DVC_PULL_TARGETS -v
    else
        dvc pull -v
    fi
fi

# ============================================================
# Extraction du blob images (conditionnelle)
# ============================================================
if [ "$SKIP_IMAGE_EXTRACT" = "true" ]; then
    echo "[entrypoint] SKIP_IMAGE_EXTRACT=true, on saute l'extraction"
else
    BLOB_PATH="/workspace/data/raw_data/images/image_train.tar.zst"
    EXTRACTED_DIR="/workspace/cache/images/image_train"
    SYMLINK_PATH="/workspace/data/raw_data/images/image_train"

    if [ -f "$BLOB_PATH" ]; then
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

            echo "[$(date +%T)] Swap du cache..."
            rm -rf "$EXTRACTED_DIR"
            mv "$TEMP_DIR/image_train" "$EXTRACTED_DIR"
            rm -rf "$TEMP_DIR"
            echo "[$(date +%T)] Swap terminé"
        fi

        rm -rf "$SYMLINK_PATH" 2>/dev/null
        ln -sfn "$EXTRACTED_DIR" "$SYMLINK_PATH"
        echo "[$(date +%T)] Symlink créé : $SYMLINK_PATH → $EXTRACTED_DIR"
        echo "[$(date +%T)] Test 5 fichiers : $(ls $SYMLINK_PATH | head -5 | tr '\n' ' ')"
    fi
fi

cd /workspace

# Cache parquet sur le volume persistant (pour réutilisation entre pods)
mkdir -p /workspace/cache/parquet_cache
mkdir -p /workspace/data
rm -rf /workspace/data/cache 2>/dev/null  # vire le dossier éphémère s'il existe
ln -sfn /workspace/cache/parquet_cache /workspace/data/cache
echo "[entrypoint] Symlink cache parquet : /workspace/data/cache → /workspace/cache/parquet_cache"

# ============================================================
# Exécution de la commande applicative
# ============================================================
echo "[entrypoint] Exécution : $*"

# NO_PROXY exhaustif : tout sauf MLflow (sur IP Tailscale)
# Seul le trafic vers 100.x.x.x (tailnet) passera par le proxy outbound
PYTHONUNBUFFERED=1
HTTP_PROXY="http://localhost:1055" \
HTTPS_PROXY="http://localhost:1055" \
NO_PROXY="localhost,127.0.0.1,*.huggingface.co,huggingface.co,*.amazonaws.com,*.cloudflarestorage.com,*.mongodb.net,*.runpod.io,download.pytorch.org,*.pytorch.org,*.pypi.org" \
"$@"
EXIT_CODE=$?


# ============================================================
# DVC push des artefacts si succès et flag actif
# ============================================================
if [ "$EXIT_CODE" -eq 0 ] && [ "${DVC_AUTO_PUSH:-false}" = "true" ]; then
    echo "[entrypoint] DVC push des nouveaux artefacts..."
    if [ -d "data/cache" ]; then
        dvc add data/cache 2>/dev/null || true
        dvc push -v || echo "[entrypoint] WARN: dvc push a échoué"
    fi
else
    echo "[entrypoint] Pas de push (exit=$EXIT_CODE, DVC_AUTO_PUSH=${DVC_AUTO_PUSH:-false})"
fi

# Le trap cleanup_and_terminate s'exécute automatiquement ici
exit $EXIT_CODE