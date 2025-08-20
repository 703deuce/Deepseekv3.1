#!/usr/bin/env python3
import os
from typing import Any, Dict, Optional, Union, List
import runpod

# Ensure a persistent HF cache on RunPod network volume
_DEFAULT_PERSISTENT_CACHE = \
    os.getenv("PERSISTENT_HF_CACHE", "/runpod-volume/hf_cache")
for _var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_var, _DEFAULT_PERSISTENT_CACHE)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Configuration via environment variables for flexibility at deploy time
MODEL_ID: str = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3")
# TensorRT-LLM FP8 configuration for H100
DTYPE: str = os.getenv("DTYPE", "fp8")  # fp8 for H100 native support
KV_CACHE_DTYPE: str = os.getenv("KV_CACHE_DTYPE", "fp8")  # fp8 KV cache on H100
MAX_MODEL_LEN: Optional[int] = int(os.getenv("MAX_MODEL_LEN", "16384")) or None
TP_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", os.getenv("TP_SIZE", "1")))
GPU_MEM_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))

_LLM_INSTANCE: Optional[Any] = None


def _build_llm():
    """Initialize TensorRT-LLM engine with FP8 support for H100."""
    try:
        # Import TensorRT-LLM modules
        from tensorrt_llm import LLM, BuildConfig, SamplingParams
        from tensorrt_llm.models import DeepSeekForCausalLM
        
        # Configure build for H100 FP8
        build_config = BuildConfig()
        build_config.precision = DTYPE  # fp8
        build_config.kv_cache_dtype = KV_CACHE_DTYPE  # fp8
        build_config.max_input_len = MAX_MODEL_LEN // 2 if MAX_MODEL_LEN else 8192
        build_config.max_output_len = MAX_MODEL_LEN // 2 if MAX_MODEL_LEN else 8192
        build_config.max_batch_size = 1
        build_config.tensor_parallel = TP_SIZE
        
        # Initialize LLM with DeepSeek-V3
        llm = LLM(
            model=MODEL_ID,
            build_config=build_config,
            trust_remote_code=True
        )
        
        return llm
        
    except ImportError:
        # Fallback to transformers if TensorRT-LLM not available
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        return {"model": model, "tokenizer": tokenizer}


def _get_llm():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = _build_llm()
    return _LLM_INSTANCE


def _normalize_prompt(event: Dict[str, Any]) -> str:
    """Accept either a raw 'prompt' string or OpenAI-style 'messages'."""
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


def _build_sampling_params(event: Dict[str, Any]) -> Dict[str, Any]:
    """Create sampling parameters from request."""
    return {
        "max_new_tokens": int(event.get("max_tokens", 512)),
        "temperature": float(event.get("temperature", 0.7)),
        "top_p": float(event.get("top_p", 0.95)),
        "top_k": int(event.get("top_k", 0)) or None,
        "repetition_penalty": 1.0 + float(event.get("frequency_penalty", 0.0)),
        "stop": event.get("stop"),
        "num_return_sequences": int(event.get("n", 1))
    }


def handler(event_or_job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless handler for TensorRT-LLM DeepSeek-V3."""
    # Support both direct event payloads and RunPod job wrappers
    event: Dict[str, Any] = event_or_job.get("input", event_or_job)

    llm = _get_llm()
    prompt = _normalize_prompt(event)
    sampling_params = _build_sampling_params(event)

    try:
        # TensorRT-LLM inference
        if hasattr(llm, 'generate'):
            # TensorRT-LLM path
            from tensorrt_llm import SamplingParams as TRTSamplingParams
            
            trt_params = TRTSamplingParams(
                max_new_tokens=sampling_params["max_new_tokens"],
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                top_k=sampling_params["top_k"]
            )
            
            outputs = llm.generate([prompt], sampling_params=trt_params)
            
            text_outputs = []
            for output in outputs:
                text_outputs.append(output.outputs[0].text)
                
        else:
            # Transformers fallback
            import torch
            from transformers import GenerationConfig
            
            model = llm["model"]
            tokenizer = llm["tokenizer"]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=sampling_params["max_new_tokens"],
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode only the new tokens
            new_tokens = outputs[:, inputs.input_ids.shape[1]:]
            text_outputs = [tokenizer.decode(tokens, skip_special_tokens=True) 
                          for tokens in new_tokens]

        return {
            "model": MODEL_ID,
            "outputs": [
                {
                    "text": text,
                    "finish_reason": "stop",
                }
                for text in text_outputs
            ],
        }

    except Exception as e:
        return {
            "model": MODEL_ID,
            "error": str(e),
            "outputs": [
                {
                    "text": f"Error: {str(e)}",
                    "finish_reason": "error",
                }
            ],
        }


def start() -> None:
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start()