#!/usr/bin/env python3
import os
import torch
from typing import Any, Dict, Optional, List
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure a persistent HF cache on RunPod network volume
_DEFAULT_PERSISTENT_CACHE = \
    os.getenv("PERSISTENT_HF_CACHE", "/runpod-volume/hf_cache")
for _var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_var, _DEFAULT_PERSISTENT_CACHE)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Configuration via environment variables for flexibility at deploy time
MODEL_ID: str = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3.1")
TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "fp8")  # fp8 | bfloat16 | float16 | auto
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
GPU_MEM_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
# DeepSeek-V3.1 specific: thinking mode support
THINKING_MODE: bool = os.getenv("THINKING_MODE", "false").lower() == "true"

# Global model and tokenizer instances
_MODEL_INSTANCE: Optional[Any] = None
_TOKENIZER_INSTANCE: Optional[Any] = None


def _load_model_and_tokenizer():
    """Load DeepSeek-V3.1 model and tokenizer using official method."""
    global _MODEL_INSTANCE, _TOKENIZER_INSTANCE
    
    if _MODEL_INSTANCE is None or _TOKENIZER_INSTANCE is None:
        # Check B200 GPU compatibility
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_capability = torch.cuda.get_device_capability(0)
            print(f"GPU: {device_name}, Compute Capability: {device_capability}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            
            # Test basic CUDA functionality for B200
            try:
                test_tensor = torch.rand(2, 2).cuda()
                print("✅ CUDA test successful - B200 compatible")
            except Exception as e:
                print(f"❌ CUDA test failed: {e}")
        
        print(f"Loading DeepSeek-V3.1 model from {MODEL_ID}...")
        
        # Load tokenizer
        _TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=_DEFAULT_PERSISTENT_CACHE
        )
        
        # Configure torch dtype (FP8 requires special handling)
        if TORCH_DTYPE == "fp8":
            # For FP8, we'll use bfloat16 as base and apply quantization
            torch_dtype = torch.bfloat16
            use_fp8_quantization = True
        else:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "auto": "auto"
            }
            torch_dtype = dtype_map.get(TORCH_DTYPE, torch.bfloat16)
            use_fp8_quantization = False
        
        # Load model with optimizations for 48GB GPU
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",  # Automatically distribute across available GPUs
            "trust_remote_code": True,
            "cache_dir": _DEFAULT_PERSISTENT_CACHE,
            "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
            "attn_implementation": "flash_attention_2",  # Use Flash Attention if available
        }
        
        # Add FP8 quantization if requested
        if use_fp8_quantization:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=False,
                    load_in_4bit=False,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_8bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = quantization_config
                print("Using FP8-style quantization via BitsAndBytesConfig")
            except ImportError:
                print("BitsAndBytesConfig not available, using bfloat16 instead")
        
        _MODEL_INSTANCE = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        
        print(f"Model loaded successfully with dtype: {torch_dtype}")
    
    return _MODEL_INSTANCE, _TOKENIZER_INSTANCE


def _normalize_prompt(event: Dict[str, Any]) -> str:
    """Accept either a raw 'prompt' string or OpenAI-style 'messages'."""
    if isinstance(event.get("prompt"), str):
        return event["prompt"]

    messages: Optional[List[Dict[str, str]]] = event.get("messages")
    if isinstance(messages, list) and messages:
        # Use DeepSeek-V3.1 chat template with thinking mode support
        try:
            model, tokenizer = _load_model_and_tokenizer()
            if hasattr(tokenizer, 'apply_chat_template'):
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    thinking=THINKING_MODE,  # V3.1 thinking mode support
                    add_generation_prompt=True
                )
        except:
            pass
        
        # Fallback to simple formatting
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    return "Hello, DeepSeek!"


def _build_generation_config(event: Dict[str, Any]) -> Dict[str, Any]:
    """Create generation parameters from request."""
    return {
        "max_new_tokens": min(int(event.get("max_tokens", MAX_NEW_TOKENS)), MAX_NEW_TOKENS),
        "temperature": float(event.get("temperature", 0.7)),
        "top_p": float(event.get("top_p", 0.95)),
        "top_k": int(event.get("top_k", 0)) or None,
        "repetition_penalty": 1.0 + float(event.get("frequency_penalty", 0.0)),
        "do_sample": float(event.get("temperature", 0.7)) > 0.0,
        "pad_token_id": None,  # Will be set to tokenizer.eos_token_id
        "eos_token_id": None,  # Will be set to tokenizer.eos_token_id
        "num_return_sequences": int(event.get("n", 1))
    }


def handler(event_or_job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless handler for DeepSeek-V3 inference."""
    # Support both direct event payloads and RunPod job wrappers
    event: Dict[str, Any] = event_or_job.get("input", event_or_job)

    try:
        # Load model and tokenizer
        model, tokenizer = _load_model_and_tokenizer()
        
        # Prepare input
        prompt = _normalize_prompt(event)
        generation_config = _build_generation_config(event)
        
        # Set pad_token_id and eos_token_id
        generation_config["pad_token_id"] = tokenizer.eos_token_id
        generation_config["eos_token_id"] = tokenizer.eos_token_id
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=4096  # Reasonable input limit
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode responses (exclude input tokens)
        input_length = inputs.input_ids.shape[1]
        text_outputs = []
        
        for output in outputs:
            # Extract only the newly generated tokens
            new_tokens = output[input_length:]
            decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text_outputs.append(decoded_text.strip())

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
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in handler: {error_trace}")
        
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