# RTX 3060 Hardware Guide

The NVIDIA RTX 3060 (12 GB GDDR6, Ampere GA106) is a popular mid-range GPU suitable for running LLMs locally.

## Specifications

- GPU: GA106 (Ampere)
- VRAM: 12 GB GDDR6
- Memory Bus: 192-bit
- Bandwidth: 360 GB/s
- CUDA Cores: 3584
- Tensor Cots (3rd gen): 112
- RT Cots (2nd gen): 28
- Base Clock: 1320 MHz
- Boost Clock: 1777 MHz
- TDP: 170 W

## Suitability for LLMs

The 12 GB VRAM allows running quantized models up to around 8B parameters in 4-bit or 5-bit quantization.
For example:
- Llama 3 8B (Q4_K_M) ~ 4.7 GB
- Mistral 7B (Q4_K_M) ~ 4.3 GB
- Phi-3 medium (14B) might be too large for full offload, but can be run with CPU offloading or smaller quantizations.

## Recommended Settings

### vLLM
- `GPU_MEMORY_UTILIZATION`: 0.90-0.95 (leave some VRAM for overhead)
- `MAX_MODEL_LEN`: 2048-4096 (depends on model and use case)
- `TENSOR_PARALLEL_SIZE`: 1 (single card)
- `dtype`: auto (or half for better speed if supported)

### llama.cpp
- Use GPU offloading with `-ngl` parameter. Start with 20-30 layers and adjust based on performance.
- Example: `./main -m model.quantized.gguf -ngl 35 -c 2048`

### General Tips
1. Monitor VRAM usage with `nvidia-smi` to avoid OOM.
2. Use lower precision (quantization) to fit larger models.
3. Consider CPU offloading for layers that don't fit in VRAM.
4. Ensure adequate cooling and power supply (170W TDP, but recommend 500W+ PSU for system).
5. Keep drivers up to date for best performance and compatibility.

## Known Issues & Workarounds

- **PCIe Bandwidth**: If using a PCIe 3.0 x16 slot, the effective bandwidth is halved compared to PCIe 4.0, which may affect multi-card scenarios or high-throughput data transfer. For single-card LLMs, impact is minimal.
- **Driver Versions**: Some users report better stability with specific driver branches (e.g., 550.xx series for CUDA 12.4). Check compatibility with your chosen inference engine.

## Example Configurations

See `models/` subdirectories for working examples:
- `models/phi-3/vllm/compose/` - vLLM setup for Phi-3-mini
- `models/mistral-7b/llama.cpp/` - llama.cpp command line for Mistral 7B

## Community

If you have success with a particular model or configuration, consider adding it to the repo or sharing in issues.
