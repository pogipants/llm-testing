# Optimization Strategies for RTX 3060

The RTX 3060 (12 GB VRAM) is a capable card for running LLMs locally. Below are various optimization techniques to maximize performance and enable larger models.

## 1. Model Quantization

Reduce VRAM usage and improve inference speed by lowering numerical precision.

### 1.1. 4-bit Quantization (bitsandbytes)
- Use `transformers` with `load_in_4bit=True`
- Requires: `bitsandbytes>=0.41.0`
- Example:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained(
      "model_name",
      load_in_4bit=True,
      device_map="auto"
  )
  ```

### 1.2. GPTQ Quantization
- Offers better accuracy than 4-bit bitsandbytes for some models
- Use `AutoGPTQ` or `optimum` pipelines
- Example:
  ```bash
  pip install auto-gptq
  python -m auto_gptq.quantize --model facebook/opt-125m --wbits 4 --groupsize 128 --output_dir quantized_model
  ```

### 1.3. GGUF (for llama.cpp)
- Convert models to GGUF format for CPU/GPU offloading
- Useful when VRAM is insufficient; layers can be offloaded to RAM
- Tools: `llama.cpp` conversion scripts

## 2. Efficient Inference Engines

### 2.1. vLLM
- High throughput and low latency with PagedAttention
- Supports continuous batching
- Installation: `pip install vllm`
- Example:
  ```bash
  vllm serve facebook/opt-125m --dtype auto --quantization awq
  ```

### 2.2. TensorRT-LLM
- NVIDIA's optimized inference library for LLMs
- Requires TensorRT and CUDA
- Steps:
  1. Convert Hugging Face model to TensorRT-LLM checkpoint
  2. Build engine with FP8 or INT8 quantization
  3. Run with `trtllm-bench`
- Best for maximum throughput on RTX 3060

### 2.3. Hugging Face TGI (Text Generation Inference)
- Optimized for deployment, supports tensor parallelism
- Good for serving multiple models

## 3. Model Offloading and CPU Hybrid

When VRAM is insufficient (e.g., for models >10B parameters):
- Use `device_map="auto"` with `transformers` to split layers between GPU and CPU
- Use `accelerate` with `cpu_offload=True`
- Note: Performance will be limited by PCIe bus speed for offloaded layers

## 4. Batch Size and Sequence Length Tuning

### 4.1. Dynamic Batching
- Engines like vLLM and TensorRT-LLM support dynamic batching to maximize GPU utilization
- Adjust `max_batch_size` based on VRAM and model size

### 4.2. Sequence Length
- Longer sequences increase VRAM usage quadratically (due to attention)
- Use sliding window or attention sinks for very long contexts if needed

## 5. Kernel Optimization

### 5.1. FlashAttention-2
- Faster and more memory-efficient attention implementation
- Requires Hopper or Ampere (RTX 30xx supports)
- Install: `pip install flash-attn --no-build-isolation`
- Ensure model uses it (some models in `transformers` have it integrated)

### 5.2. xFormers
- Memory-efficient attention operators
- Install: `pip install xformers`
- Use with `transformers` by setting `model.config.attn_implementation = "xformers"`

## 6. Compilation and Caching

### 6.1. Torch Dynamo
- PyTorch 2.0+ compilation can speed up eager mode models
- Use `torch.compile(model)` (may require adjustments for some models)

### 6.2. ONNX Runtime
- Export models to ONNX and run with ONNX Runtime GPU execution provider
- Good for static models; less flexible for generation loops

## 7. RTX 3060 Specific Tips

- **VRAM**: 12 GB allows for:
  - Up to ~13B parameters in 4-bit quantization (with some overhead)
  - Up to ~6B parameters in 8-bit quantization
  - Full precision up to ~3B parameters
- **Power Limit**: Consider increasing power limit if thermals allow (via `nvidia-smi -pl`)
- **Memory Clock**: Overclocking memory can help with bandwidth-bound operations (caution: may require stable cooling)
- **Driver**: Use latest stable driver (550+ as of 2024) for best CUDA and TensorRT support

## 8. Recommended Workflow for RTX 3060

1. **Start with quantized models** (4-bit via bitsandbytes or GPTQ) to fit larger models.
2. **Use vLLM or TensorRT-LLM** for best throughput.
3. **Enable FlashAttention-2** if the model supports it.
4. **Tune batch size** to maximize VRAM usage without OOM.
5. **Monitor** with `nvidia-smi dmon -s u` to check GPU utilization and memory.

## 9. Example: Running Llama-3-8B-Instruct on RTX 3060

```bash
# 4-bit quantization with transformers and bitsandbytes
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3-8B-Instruct-hf',
    load_in_4bit=True,
    device_map='auto',
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B-Instruct-hf')
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)
print(pipe('Explain quantum computing in simple terms')[0]['generated_text'])
"
```

For better performance, use vLLM:
```bash
vllm serve meta-llama/Llama-3-8B-Instruct-hf --dtype auto --quantization bitsandbytes --load-format bitsandbytes
```

## 10. Further Reading

- NVIDIA TensorRT-LLM Documentation: https://docs.nvidia.com/deeplearning/tensorrtllm/index.html
- vLLM Documentation: https://docs.vllm.ai/
- Hugging Face Transformers Optimization Tips: https://huggingface.co/docs/transformers/main_classes/optimization
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ

---
*This guide is a living document. Update as you discover new techniques or as software evolves.*