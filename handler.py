import os
from typing import Any, Dict, Optional, Union, List

import runpod
from vllm import LLM, SamplingParams


# Ensure a persistent HF cache on RunPod network volume
_DEFAULT_PERSISTENT_CACHE = \
    os.getenv("PERSISTENT_HF_CACHE", "/runpod-volume/hf_cache")
for _var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_var, _DEFAULT_PERSISTENT_CACHE)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Configuration via environment variables for flexibility at deploy time
MODEL_ID: str = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3")
# For FP8 preference, we try quantization="fp8" first, then fall back gracefully if unsupported
QUANTIZATION: Optional[str] = os.getenv("QUANTIZATION", "fp8")
TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "auto")  # auto | float16 | bfloat16 | float32
TP_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", os.getenv("TP_SIZE", "1")))
GPU_MEM_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
MAX_MODEL_LEN: Optional[int] = int(os.getenv("MAX_MODEL_LEN", "16384")) or None

# Optimal for 48GB GPU: FP8 weights + BF16 KV cache (most memory efficient until FP8 KV is supported)
# Note: DeepSeek MLA architecture doesn't support FP8 KV cache yet in vLLM
KV_CACHE_DTYPE: Optional[str] = os.getenv("KV_CACHE_DTYPE", "bfloat16")  # fp8 | fp16 | bfloat16 | auto


_LLM_INSTANCE: Optional[LLM] = None


def _build_llm() -> LLM:
    """Initialize the global LLM lazily, preferring FP8 settings if available.

    We attempt FP8 quantization first. If it's unsupported in the current
    environment or wheel, we fall back to a standard bfloat16/auto configuration.
    """
    init_kwargs: Dict[str, Any] = {
        "model": MODEL_ID,
        "trust_remote_code": True,
        "tensor_parallel_size": TP_SIZE,
        "gpu_memory_utilization": GPU_MEM_UTILIZATION,
        "dtype": TORCH_DTYPE,
    }

    if MAX_MODEL_LEN is not None:
        init_kwargs["max_model_len"] = MAX_MODEL_LEN

    # Prefer FP8 where possible
    if KV_CACHE_DTYPE:
        init_kwargs["kv_cache_dtype"] = KV_CACHE_DTYPE

    # Try quantization flag first (e.g., "fp8"). If it fails, retry without.
    if QUANTIZATION:
        try:
            return LLM(quantization=QUANTIZATION, **init_kwargs)
        except Exception:
            pass

    # Fallback without quantization
    # Favor bfloat16 on modern GPUs; leave override via TORCH_DTYPE env.
    if TORCH_DTYPE == "auto":
        init_kwargs["dtype"] = "bfloat16"

    return LLM(**init_kwargs)


def _get_llm() -> LLM:
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = _build_llm()
    return _LLM_INSTANCE


def _normalize_prompt(event: Dict[str, Any]) -> str:
    """Accept either a raw 'prompt' string or OpenAI-style 'messages'.

    This is a base model, so we apply a very light formatting if 'messages'
    are provided. For production, consider a proper chat template.
    """
    if isinstance(event.get("prompt"), str):
        return event["prompt"]

    messages: Optional[List[Dict[str, str]]] = event.get("messages")
    if isinstance(messages, list) and messages:
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    return "Hello, DeepSeek!"


def _build_sampling_params(event: Dict[str, Any]) -> SamplingParams:
    """Create SamplingParams from request with safe defaults."""
    max_tokens = int(event.get("max_tokens", 512))
    temperature = float(event.get("temperature", 0.7))
    top_p = float(event.get("top_p", 0.95))
    top_k = int(event.get("top_k", 0)) or None
    presence_penalty = float(event.get("presence_penalty", 0.0))
    frequency_penalty = float(event.get("frequency_penalty", 0.0))
    stop: Optional[List[str]] = event.get("stop")
    seed: Optional[int] = event.get("seed")
    n = int(event.get("n", 1))

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=stop,
        seed=seed,
        n=n,
    )


def handler(event_or_job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless handler.

    Expects an event payload like:
    {
        "prompt": "Who are you?",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "stop": ["\n\n"],
        "n": 1
    }
    or OpenAI-style messages:
    {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Introduce yourself."}
        ]
    }
    RunPod typically wraps input in a job dict: { "input": { ... } }
    """
    # Support both direct event payloads and RunPod job wrappers
    event: Dict[str, Any] = event_or_job.get("input", event_or_job)

    llm = _get_llm()

    prompt = _normalize_prompt(event)
    sampling_params = _build_sampling_params(event)

    outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)

    # vLLM returns a list of RequestOutput; each has .outputs list of candidates
    text_outputs: List[str] = []
    finish_reasons: List[Optional[str]] = []
    for req_out in outputs:
        if not req_out.outputs:
            text_outputs.append("")
            finish_reasons.append(None)
            continue
        # Collect n candidates for this prompt
        for cand in req_out.outputs:
            text_outputs.append(cand.text)
            finish_reasons.append(getattr(cand, "finish_reason", None))

    return {
        "model": MODEL_ID,
        "outputs": [
            {
                "text": text_outputs[i],
                "finish_reason": finish_reasons[i],
            }
            for i in range(len(text_outputs))
        ],
    }


def start() -> None:
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start()


