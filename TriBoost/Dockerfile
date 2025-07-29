# ───────── base image ────────────────
FROM python:3.11-slim

# ───────── system build tools ────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

# ───────── Python dependencies ───────
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ───────── application code ──────────
# Everything goes in the container root “/”
COPY service.py utils.py /

WORKDIR /              # ← current dir for uvicorn & relative paths

EXPOSE 4562
ENTRYPOINT ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "4562"]
