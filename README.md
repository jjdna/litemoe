# litemoe 🔥

**Lightweight MoE Inference on Constrained Hardware**

Run large Mixture-of-Experts (MoE) language models on limited hardware (4GB VRAM GPU, 8GB RAM) through **tiered storage** and **dynamic expert caching**.

## The Problem

MoE LLMs (Qwen3.5-35B-A3B, DeepSeek-V3/R1) have 30B–671B total parameters, far exceeding typical consumer GPU VRAM. Standard approaches require expensive hardware upgrades.

## The Solution

```
┌───────────────────────────────────────────────┐
│  Tier 1: GPU VRAM (4GB)                       │
│  → Attention layers (compute-bound)           │
│  → **Hot experts** (frequent activation)      │
├───────────────────────────────────────────────┤
│  Tier 2: CPU RAM (~8GB)                       │
│  → All expert weights (mmap hot pages)        │
├───────────────────────────────────────────────┤
│  Tier 3: SSD (unlimited)                      │
│  → Cold expert pages (kernel-managed)         │
└───────────────────────────────────────────────┘
```

## Key Ideas

| Idea | Origin | Implementation |
|------|--------|---------------|
| **Computation offloading** | KTransformers (SOSP '25) | Experts stay in CPU RAM, compute there |
| **Arithmetic intensity split** | KTransformers | Attention (high AI) → GPU; Experts (low AI) → CPU |
| **Dynamic expert caching** | **This project** | LRU+LFU hybrid: hot experts on GPU, cold on CPU |
| **SSE fallback** | llama.cpp | Run on CPUs without AVX (e.g., Pentium G4600) |
| **Expert activation profiling** | **This project** | Log which experts fire, analyze frequency patterns |

## Architecture

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────┐
│         llama.cpp Inference Engine        │
│  ┌────────┐     ┌──────────────────┐     │
│  │ Router │────▶│ Expert Scheduler │     │
│  └────────┘     └────────┬─────────┘     │
│      │                    │              │
│      ▼                    ▼              │
│  ┌────────┐     ┌──────────────────┐     │
│  │Attn ON│     │Expert ON:        │     │
│  │ GPU   │     │  Hot→GPU cache   │     │
│  │       │     │  Cold→CPU (mmap) │     │
│  └────────┘     └──────────────────┘     │
└──────────────────────────────────────────┘
    │
    ▼
Next token
```

## Status

| Component | Status | 
|-----------|--------|
| **Three-tier inference** (llama.cpp `--cpu-moe`) | ✅ Working on G4600 + GTX 1050 |
| **Partial expert GPU offload** (`--n-cpu-moe`) | ✅ Working, +14-26% speedup |
| **Dynamic expert cache** (LRU+LFU) | 🚧 Design complete, coding in progress |
| **Expert activation profiler** | 🚧 Design complete |
| **Frequency-based expert placement** | 📋 Planned |
| **Adaptive cache sizing** | 📋 Planned |

## Quick Start

```bash
# Three-tier inference (baseline — all experts on CPU)
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 18 \
  --cpu-moe \
  -c 2048 -t 4 -b 512 -ub 2048 \
  --host 0.0.0.0 --port 8111

# Partial expert GPU offload (last 4 layers)
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 20 \
  --n-cpu-moe 36 \
  -c 2048 -t 4 -b 512 -ub 2048 \
  --host 0.0.0.0 --port 8111
```

## Benchmarks (Qwen3.5-35B-A3B Q4_K_M on G4600 + GTX 1050 4GB)

| Config | VRAM | Speed | vs Baseline |
|--------|------|-------|-------------|
| `--cpu-moe` (all experts CPU) | 1,728 MiB | 0.65 tok/s | — |
| `--n-cpu-moe 36` (last 4 layers GPU) | 3,574 MiB | 0.74 tok/s | +14% |
| `--n-cpu-moe 35` (last 5 layers GPU) | 3,976 MiB | 0.82 tok/s | **+26%** |

## Project Structure

```
litemoe/
├── README.md              ← This file
├── LICENSE                ← MIT
├── docs/
│   ├── FRAMEWORK.md       ← Three-tier storage design
│   ├── EXPERT-CACHE.md    ← Dynamic expert cache design
│   └── BENCHMARKS.md      ← Full benchmark results
├── configs/               ← Ready-to-run configs
├── tools/                 ← Profiling & analysis tools
├── patches/               ← llama.cpp modifications
└── results/               ← Test logs & data
```

## References

- **KTransformers** — kvcache-ai/ktransformers (SOSP '25)
- **FastLLM** — ztzx16/fastllm
- **llama.cpp** — ggerganov/llama.cpp
