FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip setuptools wheel

RUN pip uninstall -y torch torchaudio || true

RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir runpod

CMD ["python", "-u", "handler.py"]
