FROM apache/airflow:2.7.0-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libopenblas-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    sentence-transformers \
    langchain-huggingface \
    qdrant-client \
    langchain-text-splitters \
    psycopg2-binary