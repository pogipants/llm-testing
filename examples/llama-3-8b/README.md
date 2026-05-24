# Llama-3-8B Example

This directory contains scripts and instructions for running the Llama-3-8B-Instruct model on an RTX 3060.

## Model Overview
- [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Llama-3-8B-Instruct) is an 8 billion parameter language model from Meta.
- It is designed for instruction-following and chat.
- With 4-bit quantization, it fits in the 12 GB VRAM of an RTX 3060 (using approximately 8-9 GB).
- Full precision (bf16/fp16) would require ~16 GB, which exceeds the 3060's VRAM, so quantization is essential.

## Setup

1. **Install dependencies** (if not already installed from the main repository requirements):
   ```bash
   pip install torch transformers vllm bitsandbytes accelerate
   ```

2. **Note on access**: Llama-3 models require accepting the license on Hugging Face.
   - Visit https://huggingface.co/meta-llama/Llama-3-8B-Instruct and agree to the terms.
   - Ensure you are logged in to huggingface-cli (`huggingface-cli login`) or have set your HF token as an environment variable.

3. **Download the model** (optional, the serve script will do this if needed):
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
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain the theory of relativity in simple terms"}],
    "max_tokens": 100
  }'
```

## Running with Hugging Face Transformers (for testing or lower throughput)

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = 'meta-llama/Llama-3-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,  # Essential for fitting in 12 GB VRAM
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)
print(pipe('Explain the theory of relativity in simple terms')[0]['generated_text'])
"
```

## Performance Tips for RTX 3060

- **Quantization**: 4-bit quantization via `bitsandbytes` (`load_in_4bit=True`) is necessary to fit the model in VRAM.
- **Tensor Parallel Size**: For Llama-3-8B, tensor parallel size of 1 is sufficient when using 4-bit quantization.
- **Batch Size**: For interactive chat, batch size of 1 is fine. For higher throughput, try batch sizes of 4 (may require slightly more VRAM for KV cache).
- **Memory**: Monitor VRAM usage with `nvidia-smi`. The model with 4-bit quantization should use around 8-9 GB, leaving room for KV cache and overhead.
- **Alternative**: You can also try 8-bit quantization (`load_in_8bit=True`) which uses more VRAM (~14-15 GB) but may not fit; or use GPTQ quantization for potentially better accuracy.

## Files

- `download_model.sh`: Script to download the model and tokenizer from Hugging Face (requires access).
- `serve.sh`: Script to start the vLLM server with optimized settings for RTX 3060 (4-bit quantization).
- `README.md`: This file.

## Troubleshooting

- **Out of memory**: Ensure you are using 4-bit quantization. If still OOM, reduce batch size or try reducing GPU memory allocation.
- **Access denied**: Make sure you have accepted the Llama-3 license on Hugging Face and are logged in.
- **Slow first response**: The first request includes model loading and KV cache allocation; subsequent requests will be faster.
- **Connection refused**: Ensure the server is running and accessible on port 8000.

## References

- Llama-3 model card: https://huggingface.co/meta-llama/Llama-3-8B-Instruct
- vLLM documentation: https://docs.vllm.ai/
- bitsandbytes documentation: https://github.com/TimDettmers/bitsandbytes
- Hugging Face Llama-3 usage guide: https://huggingface.co/docs/transformers/main_classes/model_llama
