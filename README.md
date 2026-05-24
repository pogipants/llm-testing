# RTX 3060 LLM Optimization Repository

This repository provides scripts, benchmarks, and guides for optimizing and deploying Large Language Models (LLMs) on NVIDIA RTX 3060 graphics cards (12 GB VRAM). It is inspired by and similar in structure to [club-3090](https://github.com/noonghunna/club-3090/).

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Examples](#examples)
- [Scripts](#scripts)
- [Benchmarks](#benchmarks)
- [Optimization Guide](#optimization-guide)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Overview
The RTX 3060 is a popular mid-range GPU with 12 GB of VRAM, making it suitable for running a variety of LLMs locally, especially when using quantization techniques. This repository aims to:
- Provide easy-to-use scripts for downloading, quantizing, and serving models.
- Offer benchmarking tools to measure latency and throughput.
- Document optimization strategies specific to the RTX 3060.
- Include example configurations for popular models like Phi-3, Llama-3, and Mistral.

## Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rtx3060-llm.git
   cd rtx3060-llm
   ```

2. **Install dependencies** (see [requirements.txt](./requirements.txt)):
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have an NVIDIA GPU with drivers installed**:
   ```bash
   nvidia-smi
   ```
   If not, see [docs/INSTALL.md](./docs/INSTALL.md) for driver and CUDA installation instructions.

4. **Choose an example model** (e.g., Phi-3-mini) and follow its README:
   ```bash
   cd examples/phi-3-mini
   ./download_model.sh
   ./serve.sh
   ```

## Documentation
- [Installation Guide](./docs/INSTALL.md) – How to install NVIDIA drivers, CUDA, and cuDNN.
- [Optimization Guide](./docs/OPTIMIZATION_GUIDE.md) – Techniques for quantization, inference engines, kernel optimizations, and more.
- [Hardware Guide](./docs/HARDWARE.md) – Adapted from club-3090, specifics for RTX 3060.
- [API References** (if any) – Auto-generated or linked.

## Examples
Each example directory contains a self-contained setup for a specific model:
- [Phi-3-mini](./examples/phi-3-mini/) – 3.8B parameter model from Microsoft, optimized for chat.
- (More examples to be added: Llama-3-8B, Mistral-7B, etc.)

Each example includes:
- `download_model.sh` – Script to download model and tokenizer.
- `serve.sh` – Script to start a vLLM server with recommended settings.
- `README.md` – Model-specific tips and performance expectations.

## Scripts
Located in the [`scripts/`](./scripts/) directory:
- [benchmark.py](./scripts/benchmark.py) – Simple latency benchmark using Hugging Face Transformers.
- [benchmark_vllm.py](./scripts/benchmark_vllm.py) – Benchmark using the vLLM inference engine.
- [launch.sh](./scripts/launch.sh) – Helper script to launch services from example directories.
- (Future scripts for quantization, conversion, etc.)

## Benchmarks
Run benchmarks to evaluate performance on your RTX 3060:
```bash
# Transformers benchmark
python scripts/benchmark.py --model microsoft/phi-2 --max-new-tokens 50

# vLLM benchmark (requires vLLM installed)
python scripts/benchmark_vllm.py --model microsoft/phi-2 --max-new-tokens 50 --batch-size 4
```
See the script's help for more options.

## Optimization Guide
See [docs/OPTIMIZATION_GUIDE.md](./docs/OPTIMIZATION_GUIDE.md) for detailed strategies including:
- 4-bit and 8-bit quantization with bitsandbytes and GPTQ.
- Using vLLM and TensorRT-LLM for high throughput.
- Enabling FlashAttention-2 and xFormers.
- Model offloading and CPU hybrid approaches.
- RTX 3060-specific tips (VRAM limits, power limits, etc.).

## Installation
If you need to install the NVIDIA driver and CUDA from scratch, follow:
[docs/INSTALL.md](./docs/INSTALL.md)

After that, install Python packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate vllm bitsandbytes
```
Adjust the CUDA version (`cu121`) to match your installed CUDA toolkit.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request to:
- Add new example models.
- Improve benchmark scripts.
- Add documentation for additional optimization techniques.
- Fix bugs or typos.

## License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by [club-3090](https://github.com/noonghunna/club-3090/).
- Thanks to the open-source communities behind Hugging Face Transformers, vLLM, TensorRT-LLM, bitsandbytes, and llama.cpp.