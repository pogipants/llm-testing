#!/usr/bin/env python3
"""
Simple latency benchmark for LLMs using Hugging Face Transformers.
Measures time to generate a fixed number of tokens.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark(model_name, prompt="Hello, my name is", max_new_tokens=50, batch_size=1):
    print(f"Benchmarking {model_name}...")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Batch size: {batch_size}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warm-up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Timed run
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for consistent timing
            pad_token_id=tokenizer.eos_token_id
        )
    end = time.time()
    
    # Calculate metrics
    total_time = end - start
    generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_second = generated_tokens / total_time
    
    print(f"\nResults:")
    print(f"Total time: {total_time:.3f} s")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Latency per token: {1000/tokens_per_second:.2f} ms")
    
    # Decode and show output (optional)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{output_text}")
    
    return {
        "total_time": total_time,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark LLM latency")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    
    benchmark(args.model, args.prompt, args.max_new_tokens, args.batch_size)