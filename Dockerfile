FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache

WORKDIR /app

# System deps including C++ compiler for vLLM/Triton
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates \
    build-essential gcc g++ && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1) + vLLM + runpod runtime
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio && \
    pip install --no-cache-dir vllm runpod

# Optional: HF token can be set at runtime via environment variables
# ARG HF_TOKEN
# ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Copy code
COPY handler.py /app/handler.py

# Ensure persistent cache directory exists at runtime
RUN mkdir -p /runpod-volume/hf_cache || true

# Default envs for FP8 on vLLM where supported
ENV MODEL_ID=deepseek-ai/DeepSeek-V3.1-Base \
    QUANTIZATION=fp8 \
    TORCH_DTYPE=auto \
    KV_CACHE_DTYPE=fp8 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.90

# RunPod serverless entry
CMD ["python3", "handler.py"]


