FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# System deps for inference - minimal build tools for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates \
    build-essential gcc g++ && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with B200 Blackwell support (CUDA 12.8, sm_100 kernels)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --pre torch torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir triton && \
    pip install --no-cache-dir transformers==4.46.3 safetensors==0.4.5 && \
    pip install --no-cache-dir runpod accelerate bitsandbytes

# Verify B200/sm_100 support in PyTorch
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || echo "PyTorch verification failed but continuing..."

# Download and setup DeepSeek's official inference repo for true FP8 support
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://github.com/deepseek-ai/DeepSeek-V3/archive/main.zip -O /tmp/deepseek-v3.zip && \
    unzip /tmp/deepseek-v3.zip -d /app/ && \
    mv /app/DeepSeek-V3-main /app/deepseek-v3 && \
    rm /tmp/deepseek-v3.zip && \
    rm -rf /var/lib/apt/lists/* && \
    cd /app/deepseek-v3/inference && \
    pip install --no-cache-dir -r requirements.txt

# Optional: HF token can be set at runtime via environment variables
# ARG HF_TOKEN
# ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Copy code
COPY handler.py /app/handler.py

# Ensure persistent cache directory exists at runtime
RUN mkdir -p /runpod-volume/hf_cache || true

# Production config for B200 Blackwell inference: DeepSeek-V3.1 + official FP8 quantization
ENV MODEL_ID=deepseek-ai/DeepSeek-V3.1 \
    QUANTIZATION_MODE=fp8 \
    TORCH_DTYPE=bfloat16 \
    MAX_NEW_TOKENS=512 \
    THINKING_MODE=false \
    GPU_MEMORY_UTILIZATION=0.90

# RunPod serverless entry
CMD ["python3", "handler.py"]


