#!/usr/bin/env python3
"""
Benchmark script for vLLM engine.
Measures latency and throughput for text generation.
"""

import time
import argparse
from vllm import LLM, SamplingParams

def benchmark_vllm(model_name, prompt="Hello, my name is", max_new_tokens=50, 
                   batch_size=1, tensor_parallel_size=1, dtype="auto"):
    print(f"Benchmarking vLLM with model: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Batch size: {batch_size}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Dtype: {dtype}")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for deterministic timing
        max_tokens=max_new_tokens,
        stop_token_ids=None
    )
    
    # Initialize the LLM engine
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True  # required for some models
    )
    
    # Prepare prompts (batch_size copies of the same prompt)
    prompts = [prompt] * batch_size
    
    # Warm-up
    _ = llm.generate(prompts, sampling_params)
    
    # Timed run
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    
    # Calculate metrics
    total_time = end - start
    # Count generated tokens across all sequences
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = generated_tokens / total_time
    
    print(f"\nResults:")
    print(f"Total time: {total_time:.3f} s")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Latency per token: {1000/tokens_per_second:.2f} ms")
    print(f"Throughput (sequences/sec): {batch_size/total_time:.2f}")
    
    # Show first output as example
    if outputs:
        print(f"\nExample output:\n{outputs[0].outputs[0].text}")
    
    return {
        "total_time": total_time,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "throughput_seq_per_sec": batch_size/total_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM latency with vLLM")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (GPU count)")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type (auto, half, float16, bfloat16, float32)")
    args = parser.parse_args()
    
    benchmark_vllm(
        args.model,
        args.prompt,
        args.max_new_tokens,
        args.batch_size,
        args.tensor_parallel_size,
        args.dtype
    )