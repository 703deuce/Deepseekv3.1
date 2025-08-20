FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache

WORKDIR /app

# System deps including C++ compiler, CUDA dev tools, and MPI for TensorRT-LLM
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates \
    build-essential gcc g++ \
    cuda-nvcc-12-1 cuda-cudart-dev-12-1 \
    libopenmpi-dev openmpi-bin && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch + DeepSeek-Infer dependencies for FP8 support + runpod runtime
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 triton==3.0.0 transformers==4.46.3 safetensors==0.4.5 && \
    pip install --no-cache-dir runpod accelerate

# Optional: HF token can be set at runtime via environment variables
# ARG HF_TOKEN
# ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Copy code
COPY handler.py /app/handler.py

# Ensure persistent cache directory exists at runtime
RUN mkdir -p /runpod-volume/hf_cache || true

# Config for 48GB GPU: DeepSeek-Infer with BF16 precision + Flash Attention
ENV MODEL_ID=deepseek-ai/DeepSeek-V3 \
    TORCH_DTYPE=bfloat16 \
    MAX_NEW_TOKENS=512 \
    GPU_MEMORY_UTILIZATION=0.90

# RunPod serverless entry
CMD ["python3", "handler.py"]


