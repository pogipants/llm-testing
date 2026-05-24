# Phi-3-mini Example

This directory contains scripts and instructions for running the Phi-3-mini model on an RTX 3060.

## Model Overview
- [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) is a 3.8 billion parameter language model from Microsoft.
- It is designed for instruction-following and chat.
- With 4-bit quantization, it fits comfortably in the 12 GB VRAM of an RTX 3060.

## Setup

1. **Install dependencies** (if not already installed from the main repository requirements):
   ```bash
   pip install torch transformers vllm bitsandbytes accelerate
   ```

2. **Download the model** (optional, the serve script will do this if needed):
   ```bash
   ./download_model.sh
   ```

## Running with vLLM (Recommended for best performance)

```bash
./serve.sh
```

This will start an OpenAI-compatible API server at `http://localhost:8000`.

### Example request:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phi-3-mini-4k-instruct",
    "messages": [{"role": "user", "content": "Explain the theory of relativity in simple terms"}],
    "max_tokens": 100
  }'
```

## Running with Hugging Face Transformers (for testing or lower throughput)

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)
print(pipe('Explain the theory of relativity in simple terms')[0]['generated_text'])
"
```

## Performance Tips for RTX 3060

- **Quantization**: Use 4-bit quantization via `bitsandbytes` (as in the vLLM serve command) to fit the model in VRAM.
- **Tensor Parallel Size**: For Phi-3-mini, tensor parallel size of 1 is sufficient (it fits on a single GPU).
- **Batch Size**: Adjust based on your workload. For interactive chat, batch size of 1 is fine. For higher throughput, try batch sizes of 4 or 8.
- **Memory**: Monitor VRAM usage with `nvidia-smi`. The model with 4-bit quantization should use around 8-9 GB, leaving room for KV cache and overhead.

## Files

- `download_model.sh`: Script to download the model and tokenizer from Hugging Face.
- `serve.sh`: Script to start the vLLM server with optimized settings for RTX 3060.
- `benchmark.py`: (in the main scripts directory) Can be used to benchmark latency and throughput.

## Troubleshooting

- **Out of memory**: Try reducing the batch size or ensure you are using 4-bit quantization.
- **Slow first response**: The first request includes model loading and KV cache allocation; subsequent requests will be faster.
- **Connection refused**: Ensure the server is running and accessible on port 8000.

## References

- Phi-3 model card: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- vLLM documentation: https://docs.vllm.ai/
- bitsandbytes documentation: https://github.com/TimDettmers/bitsandbytes
