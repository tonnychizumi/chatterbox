FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir .

RUN pip install --no-cache-dir runpod

ENV TRANSFORMERS_ATTN_IMPLEMENTATION=eager

CMD ["python", "-u", "handler.py"]
