FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/hub

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --ignore-installed blinker && \
    pip install -r requirements.txt

COPY server.py session.py asr_helpers.py ./

EXPOSE 8765

# Mount /cache/huggingface as a volume in production to avoid re-downloading the model
CMD ["python", "server.py"]