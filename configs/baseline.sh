#!/bin/bash
# Config A: Baseline — All MoE experts on CPU
# VRAM: ~1,728 MiB | Speed: ~0.65 tok/s
llama-server \
  -m /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 18 \
  --cpu-moe \
  -c 2048 \
  -t 4 \
  -b 512 \
  -ub 2048 \
  --host 0.0.0.0 \
  --port 8111
