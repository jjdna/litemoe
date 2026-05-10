#!/bin/bash
# Config B: Partial Expert GPU Offload (last 4 layers)
# VRAM: ~3,574 MiB (88%) | Speed: ~0.74 tok/s (+14%)
llama-server \
  -m /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 20 \
  --n-cpu-moe 36 \
  -c 2048 \
  -t 4 \
  -b 512 \
  -ub 2048 \
  --host 0.0.0.0 \
  --port 8111
