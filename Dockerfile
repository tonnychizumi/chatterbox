FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install .
RUN pip install runpod

CMD ["python", "-u", "handler.py"]
