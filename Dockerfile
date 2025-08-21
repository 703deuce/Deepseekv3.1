FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# System deps for inference - minimal build tools for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates \
    build-essential gcc g++ && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch 2.7.0+ with optimized B200/Blackwell kernels for CUDA 13.0
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
        "torch>=2.7.0" "torchvision>=0.20.0" "torchaudio>=2.7.0" triton && \
    pip install --no-cache-dir transformers==4.46.3 safetensors==0.4.5 && \
    pip install --no-cache-dir runpod accelerate bitsandbytes

# Optional: HF token can be set at runtime via environment variables
# ARG HF_TOKEN
# ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Copy code
COPY handler.py /app/handler.py

# Ensure persistent cache directory exists at runtime
RUN mkdir -p /runpod-volume/hf_cache || true

# Production config for B200 Blackwell inference: DeepSeek-V3.1 + CUDA 13.0 + optimized sm_100
ENV MODEL_ID=deepseek-ai/DeepSeek-V3.1 \
    TORCH_DTYPE=fp8 \
    MAX_NEW_TOKENS=512 \
    THINKING_MODE=false \
    GPU_MEMORY_UTILIZATION=0.90

# RunPod serverless entry
CMD ["python3", "handler.py"]


