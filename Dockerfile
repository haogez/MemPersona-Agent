FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install -U pip && python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir vllm huggingface_hub

COPY . .
ENV PYTHONPATH=/app/src

CMD ["bash","-lc","uvicorn mem_persona_agent.api.main:app --host 0.0.0.0 --port 8000"]
