#!/usr/bin/env python3
"""
Test script for DeepSeek V3.1 RunPod Serverless API
Shows how to call the endpoint from another Python application (Flask, FastAPI, etc.)
"""

import os
import time
import requests
from typing import Dict, Any, Optional, List


class DeepSeekAPIClient:
    """Client for calling DeepSeek V3.1 RunPod Serverless endpoint."""
    
    def __init__(self, api_key: str, endpoint_id: str, base_url: str = "https://api.runpod.ai"):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_sync(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Synchronous generation - blocks until completion."""
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "stop": kwargs.get("stop"),
                "seed": kwargs.get("seed"),
                "n": kwargs.get("n", 1)
            }
        }
        
        url = f"{self.base_url}/v2/{self.endpoint_id}/runsync"
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generation - returns job ID for polling."""
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "stop": kwargs.get("stop"),
                "seed": kwargs.get("seed"),
                "n": kwargs.get("n", 1)
            }
        }
        
        url = f"{self.base_url}/v2/{self.endpoint_id}/run"
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()["id"]
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of async job."""
        url = f"{self.base_url}/v2/{self.endpoint_id}/status/{job_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict[str, Any]:
        """Poll async job until completion."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_status(job_id)
            
            if status["status"] == "COMPLETED":
                return status
            elif status["status"] == "FAILED":
                raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate using OpenAI-style messages format."""
        payload = {
            "input": {
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "stop": kwargs.get("stop"),
                "seed": kwargs.get("seed"),
                "n": kwargs.get("n", 1)
            }
        }
        
        url = f"{self.base_url}/v2/{self.endpoint_id}/runsync"
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()


def test_api():
    """Test the DeepSeek API with various examples."""
    
    # Get credentials from environment variables
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    
    if not api_key or not endpoint_id:
        print("❌ Error: Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables")
        print("Example:")
        print('export RUNPOD_API_KEY="your_api_key_here"')
        print('export RUNPOD_ENDPOINT_ID="your_endpoint_id_here"')
        return
    
    client = DeepSeekAPIClient(api_key, endpoint_id)
    
    print("🚀 Testing DeepSeek V3.1 API...")
    print(f"📡 Endpoint: {endpoint_id}")
    print("-" * 50)
    
    # Test 1: Simple prompt (synchronous)
    print("\n📝 Test 1: Simple prompt (sync)")
    try:
        result = client.generate_sync(
            prompt="Explain the benefits of FP8 quantization for large language models in one paragraph.",
            max_tokens=256,
            temperature=0.2
        )
        
        if "output" in result and "outputs" in result["output"]:
            text = result["output"]["outputs"][0]["text"]
            print(f"✅ Response: {text[:200]}...")
        else:
            print(f"⚠️  Unexpected response format: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Async generation
    print("\n🔄 Test 2: Async generation")
    try:
        job_id = client.generate_async(
            prompt="Write a short Python function to calculate fibonacci numbers.",
            max_tokens=200,
            temperature=0.3
        )
        print(f"📋 Job ID: {job_id}")
        
        result = client.wait_for_completion(job_id)
        if "output" in result and "outputs" in result["output"]:
            text = result["output"]["outputs"][0]["text"]
            print(f"✅ Response: {text[:200]}...")
        else:
            print(f"⚠️  Unexpected response format: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Messages format (chat-like)
    print("\n💬 Test 3: Messages format")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What is the difference between FP8 and FP16 precision?"}
        ]
        
        result = client.generate_with_messages(
            messages=messages,
            max_tokens=300,
            temperature=0.4
        )
        
        if "output" in result and "outputs" in result["output"]:
            text = result["output"]["outputs"][0]["text"]
            print(f"✅ Response: {text[:200]}...")
        else:
            print(f"⚠️  Unexpected response format: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Multiple completions
    print("\n🎲 Test 4: Multiple completions (n=2)")
    try:
        result = client.generate_sync(
            prompt="Complete this sentence: 'The future of AI is'",
            max_tokens=50,
            temperature=0.8,
            n=2
        )
        
        if "output" in result and "outputs" in result["output"]:
            outputs = result["output"]["outputs"]
            for i, output in enumerate(outputs):
                print(f"✅ Completion {i+1}: {output['text']}")
        else:
            print(f"⚠️  Unexpected response format: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Testing completed!")


# Example usage in a Flask app
def flask_example():
    """Example of how to use in a Flask backend."""
    
    # This would be in your Flask app
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Initialize client (do this once, not per request)
    client = DeepSeekAPIClient(
        api_key=os.getenv("RUNPOD_API_KEY"),
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID")
    )
    
    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.json
        prompt = data.get("prompt", "")
        
        try:
            result = client.generate_sync(
                prompt=prompt,
                max_tokens=data.get("max_tokens", 512),
                temperature=data.get("temperature", 0.7)
            )
            
            # Extract the text from the response
            if "output" in result and "outputs" in result["output"]:
                text = result["output"]["outputs"][0]["text"]
                return jsonify({"success": True, "text": text})
            else:
                return jsonify({"success": False, "error": "Unexpected response format"})
                
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    return app


if __name__ == "__main__":
    test_api()
