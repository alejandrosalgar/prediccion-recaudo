# Imagen base Python
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (por si pandas/sklearn las necesitan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python (solo API)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código de la API
COPY api.py .

# Cloud Run inyecta PORT (por defecto 8080)
ENV PORT=8080
EXPOSE 8080

# Gunicorn: app es el nombre del módulo (api.py) y app es la variable Flask
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 60 api:app
