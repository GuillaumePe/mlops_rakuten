FROM python:3.10-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie requirements
COPY docker/api/requirements.api.txt ./requirements.txt

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


# Copie du code source
COPY src/ ./src/
COPY docker/api/requirements.api.txt ./requirements.txt

# Exposition du port FastAPI
EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]