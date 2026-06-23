# Base : PyTorch officielle CUDA 12.1, Python 3.10
# Versions à ajuster selon les évolutions de PyTorch / CUDA
# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime  

# === Métadonnées ===
LABEL maintainer="GP"
LABEL description="Training image and text for MLOps Rakuten project (extraction + training)"

# === Variables d'environnement ===
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false

# === Outils système ===
# git : pour le code et DVC qui peut faire des appels git
# curl, ca-certificates : pour télécharger des binaires
# build-essential : nécessaire pour certaines deps Python compilées
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential iptables iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://tailscale.com/install.sh | sh \
    && mkdir -p /var/lib/tailscale /var/run/tailscale \
    && tailscale --version
# === Workspace ===
WORKDIR /workspace

# === Dependencies Python applicatives ===
# On copie d'abord les requirements pour bénéficier du cache Docker.
# Si requirements ne change pas, cette couche est cachée et on évite
# de re-télécharger tous les packages à chaque rebuild.
COPY docker/trainer/requirements.trainer.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -v -r requirements.txt

# === Code applicatif ===
# Copié après les deps pour minimiser les rebuilds quand seul le code change.
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY .dvc/config /workspace/.dvc/config
# === Fichiers dvc ===
COPY data/ /workspace/data/
# === Entrypoint ===
# Le script gère le cycle DVC pull → command → DVC push
COPY docker/trainer/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

COPY docker/trainer/mongo_tunnel.py /workspace/mongo_tunnel.py

# === Sanity check au build ===
# Vérifie qu'on peut au moins importer les modules clés
# + valide la présence de sm_120 dans les architectures CUDA compilées
# === Sanity check au build ===
COPY docker/trainer/sanity_check.py /workspace/sanity_check.py
RUN cd /workspace && python sanity_check.py \
    && tailscale --version \
    && echo "Image trainer OK"

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["echo", "No command provided"]