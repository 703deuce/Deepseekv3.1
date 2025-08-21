#!/usr/bin/env python3
import os
import sys
import torch
from typing import Any, Dict, Optional, List
import runpod

# Setup DeepSeek-V3 official inference path (use network volume for persistence)
DEEPSEEK_REPO_PATH = "/runpod-volume/deepseek-v3"

# Copy repo to network volume if not already there (for persistence across workers)
if not os.path.exists(DEEPSEEK_REPO_PATH) and os.path.exists("/app/deepseek-v3"):
    import shutil
    print("Copying DeepSeek repo to network volume for persistence...")
    shutil.copytree("/app/deepseek-v3", DEEPSEEK_REPO_PATH)
    print("✅ DeepSeek repo copied to network volume")

# Fallback to app directory if network volume copy failed
if not os.path.exists(DEEPSEEK_REPO_PATH):
    DEEPSEEK_REPO_PATH = "/app/deepseek-v3"
    print("Using DeepSeek repo from app directory")

sys.path.append(DEEPSEEK_REPO_PATH)
sys.path.append(f"{DEEPSEEK_REPO_PATH}/inference")

# Import DeepSeek's official FP8 inference modules
try:
    from inference.models import DeepSeekV3ForCausalLM
    from inference.tokenizer import get_tokenizer
    from inference.generation import generate_response
    DEEPSEEK_AVAILABLE = True
except ImportError:
    # Fallback to transformers if DeepSeek modules not available
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DEEPSEEK_AVAILABLE = False
    print("Warning: DeepSeek official inference not available, using Transformers")

# Ensure a persistent HF cache on RunPod network volume
_DEFAULT_PERSISTENT_CACHE = os.getenv("PERSISTENT_HF_CACHE", "/runpod-volume/hf_cache")
for _var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_var, _DEFAULT_PERSISTENT_CACHE)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Configuration via environment variables for flexibility at deploy time
MODEL_ID: str = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3.1")
QUANTIZATION_MODE: str = os.getenv("QUANTIZATION_MODE", "fp8")  # fp8 | none | int8
TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "bfloat16")  # bfloat16 | float16 | auto
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
GPU_MEM_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
# DeepSeek-V3.1 specific: thinking mode support
THINKING_MODE: bool = os.getenv("THINKING_MODE", "false").lower() == "true"

# Global model and tokenizer instances
_MODEL_INSTANCE: Optional[Any] = None
_TOKENIZER_INSTANCE: Optional[Any] = None


def _load_model_and_tokenizer():
    """Load DeepSeek-V3.1 model using official FP8 inference method."""
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
        
        if DEEPSEEK_AVAILABLE:
            print(f"Loading DeepSeek-V3.1 model with {QUANTIZATION_MODE} quantization using official inference...")
            
            # Load tokenizer using DeepSeek's official method
            _TOKENIZER_INSTANCE = get_tokenizer(
                model_path=MODEL_ID,
                cache_dir=_DEFAULT_PERSISTENT_CACHE
            )
            
            # Load model using DeepSeek's official FP8 method
            _MODEL_INSTANCE = DeepSeekV3ForCausalLM.from_pretrained(
                model_path=MODEL_ID,
                quantization_mode=QUANTIZATION_MODE,
                torch_dtype=TORCH_DTYPE,
                cache_dir=_DEFAULT_PERSISTENT_CACHE,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"✅ DeepSeek-V3.1 loaded successfully with {QUANTIZATION_MODE} quantization")
        else:
            # Fallback to Transformers (without FP8)
            print(f"Loading DeepSeek-V3.1 model using Transformers fallback...")
            
            _TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                cache_dir=_DEFAULT_PERSISTENT_CACHE
            )
            
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "auto": "auto"
            }
            torch_dtype = dtype_map.get(TORCH_DTYPE, torch.bfloat16)
            
            _MODEL_INSTANCE = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=_DEFAULT_PERSISTENT_CACHE,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )
            
            print(f"✅ DeepSeek-V3.1 loaded with Transformers fallback")
    
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
    
    # Default prompt
    return "Hello, DeepSeek!"


def _build_generation_config(event: Dict[str, Any]) -> Dict[str, Any]:
    """Build generation configuration from event parameters."""
    return {
        "max_new_tokens": int(event.get("max_tokens", MAX_NEW_TOKENS)),
        "temperature": float(event.get("temperature", 0.7)),
        "top_p": float(event.get("top_p", 0.95)),
        "top_k": int(event.get("top_k", 0)) or None,
        "do_sample": float(event.get("temperature", 0.7)) > 0.0,
        "num_return_sequences": int(event.get("n", 1)),
    }


def handler(event_or_job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless handler for DeepSeek-V3.1 FP8 inference."""
    # Support both direct event payloads and RunPod job wrappers
    event: Dict[str, Any] = event_or_job.get("input", event_or_job)

    try:
        # Load model and tokenizer
        model, tokenizer = _load_model_and_tokenizer()
        
        # Prepare input
        prompt = _normalize_prompt(event)
        generation_config = _build_generation_config(event)
        
        if DEEPSEEK_AVAILABLE:
            # Use DeepSeek's official generation method
            outputs = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                **generation_config
            )
            
            # Format response for RunPod
            text_outputs = []
            if isinstance(outputs, list):
                for output in outputs:
                    text_outputs.append({
                        "text": output,
                        "finish_reason": "stop"
                    })
            else:
                text_outputs.append({
                    "text": str(outputs),
                    "finish_reason": "stop"
                })
        else:
            # Fallback to standard Transformers generation
            generation_config["pad_token_id"] = tokenizer.eos_token_id
            generation_config["eos_token_id"] = tokenizer.eos_token_id
            
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=4096
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
                generated_tokens = output[input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                text_outputs.append({
                    "text": generated_text,
                    "finish_reason": "stop"
                })

        return {
            "model": MODEL_ID,
            "quantization": QUANTIZATION_MODE if DEEPSEEK_AVAILABLE else "none",
            "outputs": text_outputs,
        }

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "FAILED"
        }


def start() -> None:
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start()