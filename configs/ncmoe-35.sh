#!/bin/bash
# Config C: Aggressive GPU Offload (last 5 layers)
# VRAM: ~3,976 MiB (98.4%) | Speed: ~0.82 tok/s (+26%)
# WARNING: Very tight VRAM margin — may OOM on some models
llama-server \
  -m /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 18 \
  --n-cpu-moe 35 \
  -c 2048 \
  -t 4 \
  -b 512 \
  -ub 2048 \
  --host 0.0.0.0 \
  --port 8111
