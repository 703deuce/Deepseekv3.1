#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import tempfile
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

# Ensure a persistent HF cache on RunPod network volume
_DEFAULT_PERSISTENT_CACHE = os.getenv("PERSISTENT_HF_CACHE", "/runpod-volume/hf_cache")
for _var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_var, _DEFAULT_PERSISTENT_CACHE)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Configuration via environment variables for flexibility at deploy time
MODEL_ID: str = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3.1")
QUANTIZATION_MODE: str = os.getenv("QUANTIZATION_MODE", "fp8")  # fp8 | none | int8
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
THINKING_MODE: bool = os.getenv("THINKING_MODE", "false").lower() == "true"

# Global paths for converted model
_CONVERTED_MODEL_PATH = "/runpod-volume/deepseek-v3-converted"
_MODEL_CONVERTED = False


def _check_deepseek_scripts():
    """Check if DeepSeek conversion scripts are available."""
    convert_script = os.path.join(DEEPSEEK_REPO_PATH, "convert.py")
    generate_script = os.path.join(DEEPSEEK_REPO_PATH, "generate.py")
    
    print(f"Checking for DeepSeek scripts...")
    print(f"DeepSeek repo path: {DEEPSEEK_REPO_PATH}")
    print(f"Directory exists: {os.path.exists(DEEPSEEK_REPO_PATH)}")
    
    if os.path.exists(DEEPSEEK_REPO_PATH):
        files = os.listdir(DEEPSEEK_REPO_PATH)
        print(f"Files in repo: {files[:10]}...")  # Show first 10 files
    
    convert_exists = os.path.exists(convert_script)
    generate_exists = os.path.exists(generate_script)
    
    print(f"convert.py exists: {convert_exists}")
    print(f"generate.py exists: {generate_exists}")
    
    return convert_exists and generate_exists


def _convert_model_to_fp8():
    """Convert DeepSeek model to FP8 using official convert.py script if available."""
    global _MODEL_CONVERTED
    
    if _MODEL_CONVERTED or os.path.exists(_CONVERTED_MODEL_PATH):
        print("✅ Model already converted")
        _MODEL_CONVERTED = True
        return True
    
    # Check if scripts are available
    if not _check_deepseek_scripts():
        print("⚠️ DeepSeek conversion scripts not found, will use Transformers with 8-bit quantization")
        return False
    
    print(f"Converting DeepSeek-V3.1 to FP8 quantization...")
    
    try:
        # Change to DeepSeek repo directory
        original_cwd = os.getcwd()
        os.chdir(DEEPSEEK_REPO_PATH)
        
        # Run the conversion script
        convert_cmd = [
            "python", "convert.py",
            "--hf-ckpt-path", MODEL_ID,
            "--save-path", _CONVERTED_MODEL_PATH,
            "--quant-mode", QUANTIZATION_MODE,
            "--cache-dir", _DEFAULT_PERSISTENT_CACHE
        ]
        
        print(f"Running conversion: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print("✅ Model converted to FP8 successfully")
            _MODEL_CONVERTED = True
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def _generate_with_deepseek(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate text using DeepSeek's official generate.py script with FP8."""
    
    try:
        # Change to DeepSeek repo directory
        original_cwd = os.getcwd()
        os.chdir(DEEPSEEK_REPO_PATH)
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            input_file = f.name
        
        # Create temporary output file
        output_file = tempfile.mktemp(suffix='.txt')
        
        # Run the generation script
        generate_cmd = [
            "torchrun", "--nproc_per_node=1", "generate.py",
            "--ckpt-path", _CONVERTED_MODEL_PATH,
            "--input-file", input_file,
            "--output-file", output_file,
            "--max-new-tokens", str(max_new_tokens),
            "--temperature", str(temperature),
            "--quant-mode", QUANTIZATION_MODE
        ]
        
        if THINKING_MODE:
            generate_cmd.extend(["--thinking-mode"])
        
        print(f"Generating with DeepSeek FP8: {' '.join(generate_cmd)}")
        result = subprocess.run(generate_cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            # Read the generated output
            with open(output_file, 'r') as f:
                generated_text = f.read().strip()
            
            # Clean up temporary files
            os.unlink(input_file)
            os.unlink(output_file)
            
            return generated_text
        else:
            print(f"❌ Generation failed: {result.stderr}")
            return f"Generation failed: {result.stderr}"
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return f"Generation error: {e}"
    finally:
        os.chdir(original_cwd)


def _normalize_prompt(event: Dict[str, Any]) -> str:
    """Accept either a raw 'prompt' string or OpenAI-style 'messages'."""
    if isinstance(event.get("prompt"), str):
        return event["prompt"]

    messages: Optional[List[Dict[str, str]]] = event.get("messages")
    if isinstance(messages, list) and messages:
        # Format messages for DeepSeek
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)
    
    # Default prompt
    return "Hello, DeepSeek!"


def _generate_with_transformers(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Fallback generation using Transformers with 8-bit quantization."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        print("Loading model with Transformers + 8-bit quantization...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=_DEFAULT_PERSISTENT_CACHE
        )
        
        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        
        # Load model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=_DEFAULT_PERSISTENT_CACHE,
            low_cpu_mem_usage=True
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text
        
    except Exception as e:
        return f"Transformers generation failed: {e}"


def handler(event_or_job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless handler for DeepSeek-V3.1 inference with FP8 or 8-bit fallback."""
    # Support both direct event payloads and RunPod job wrappers
    event: Dict[str, Any] = event_or_job.get("input", event_or_job)

    try:
        # Prepare input
        prompt = _normalize_prompt(event)
        max_new_tokens = int(event.get("max_tokens", MAX_NEW_TOKENS))
        temperature = float(event.get("temperature", 0.7))
        
        # Try to convert model to FP8 and use official scripts
        if _convert_model_to_fp8():
            print("Using DeepSeek official FP8 generation...")
            generated_text = _generate_with_deepseek(prompt, max_new_tokens, temperature)
            quantization_used = "fp8_official"
        else:
            print("Using Transformers with 8-bit quantization fallback...")
            generated_text = _generate_with_transformers(prompt, max_new_tokens, temperature)
            quantization_used = "8bit_transformers"
        
        # Format response for RunPod
        return {
            "model": MODEL_ID,
            "quantization": quantization_used,
            "outputs": [{
                "text": generated_text,
                "finish_reason": "stop"
            }],
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