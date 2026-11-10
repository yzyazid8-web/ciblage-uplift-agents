# syntax=docker/dockerfile:1

# Python figé sur la version de développement.
FROM python:3.12.2-slim

# libgomp1 : parallélisme d'XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (bonne pratique conteneur)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR $HOME/app

# Dépendances d'exécution figées (couche mise en cache tant que requirements.txt ne change pas)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# Code du projet, installé en mode éditable : il reste dans /home/user/app, donc les chemins du
# projet (data/, reports/) pointent au bon endroit, comme en développement
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user --no-deps -e .

# Données Hillstrom téléchargées À LA CONSTRUCTION : le conteneur démarre sans dépendre du site source
RUN python -c "from ciblage.data import hillstrom; hillstrom.download(); print('Hillstrom :', len(hillstrom.load()), 'clients')"

# Modèles entraînés et résultats pré-calculés À LA CONSTRUCTION : le conteneur démarre sans entraîner
# (réveil plus rapide quand l'hébergeur le rallume, et moins de calcul sur un petit processeur)
RUN python -m ciblage.serve.planner

COPY --chown=user app.py ./

# Tableau de bord marimo (conversation avec l'agent + graphiques). Pour l'API :
#   docker run -p 8000:8000 <image> uvicorn ciblage.serve.api:app --host 0.0.0.0 --port 8000
# Clé Groq (facultative : sans elle, le formulaire et l'API /plan fonctionnent, l'agent non) :
#   docker run -p 8080:8080 -e GROQ_API_KEY=gsk_... <image>
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080', timeout=4).status==200 else 1)"
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0", "-p", "8080", "--headless"]
