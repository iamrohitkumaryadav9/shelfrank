# API image: no torch, no scikit-learn, fp32 ONNX query encoder downloaded at build time.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY shelfrank shelfrank
COPY configs configs
COPY app_data app_data
ENV SHELFRANK_MODELS=/srv/models PYTHONPATH=/srv
RUN python -c "from pathlib import Path; from shelfrank.app_support.encoder import fetch_model; fetch_model(Path('/srv/models'))"
RUN useradd --system runner && chown -R runner /srv
USER runner
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "shelfrank.app_support.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
