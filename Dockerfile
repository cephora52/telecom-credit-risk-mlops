# Image de base : Python 3.12 léger
FROM python:3.12-slim

# Évite les fichiers .pyc et affiche les logs Python immédiatement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définit le dossier de travail à l’intérieur du conteneur
WORKDIR /app

# Installe une bibliothèque système nécessaire à XGBoost
# puis supprime les fichiers temporaires d’installation
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie la liste des dépendances de l’API dans le conteneur
COPY requirements-api.txt .

# Installe les dépendances Python de l’API
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-api.txt

# Copie le code de l’application dans le conteneur
COPY src/telecom_mlops ./src/telecom_mlops

# Copie le modèle XGBoost utilisé par l’API
COPY models/default_risk_model.joblib ./models/default_risk_model.joblib

# Indique que l’application écoute sur le port 8000
EXPOSE 8000

# Démarre le serveur API lorsque le conteneur se lance
# --app-dir src permet à Python de trouver le paquet telecom_mlops
# 0.0.0.0 rend l’API accessible depuis l’extérieur du conteneur
CMD ["uvicorn", "telecom_mlops.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
