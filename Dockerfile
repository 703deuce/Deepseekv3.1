FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache \
    HF_HOME=/app/hf_cache

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1) + vLLM + runpod runtime
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio && \
    pip install vllm==0.5.4.post1 runpod==1.7.1

# Optional: pre-auth to HF via build-arg (not recommended) or mount at runtime
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Copy code
COPY handler.py /app/handler.py

# Default envs for FP8 on vLLM where supported
ENV MODEL_ID=deepseek-ai/DeepSeek-V3.1-Base \
    QUANTIZATION=fp8 \
    TORCH_DTYPE=auto \
    KV_CACHE_DTYPE=fp8 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.90

# RunPod serverless entry
CMD ["python", "handler.py"]


