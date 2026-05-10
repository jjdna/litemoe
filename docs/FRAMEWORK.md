# Three-Tier MoE Inference Framework

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Tier 1: GPU VRAM                   │
│  GTX 1050 · 4GB · ~1-2 TB/s bandwidth                │
│  ┌─────────────────────────────────────────────┐    │
│  │ Attention layers (18/40 layers)             │    │
│  │ Shared expert FFN (when on GPU)             │    │
│  │ Embedding + output projection               │    │
│  │ Hot expert cache (dynamic)                  │    │
│  └─────────────────────────────────────────────┘    │
│  Purpose: Compute-intensive components (AI 512+)    │
├─────────────────────────────────────────────────────┤
│                    Tier 2: CPU RAM                    │
│  DDR4 · 5.5GB usable · ~19 GB/s bandwidth           │
│  ┌─────────────────────────────────────────────┐    │
│  │ All MoE expert weights (mmap hot pages)     │    │
│  │ Remaining attention layer metadata          │    │
│  └─────────────────────────────────────────────┘    │
│  Purpose: Memory-intensive components (AI 0.075)    │
├─────────────────────────────────────────────────────┤
│                    Tier 3: SSD (mmap)                │
│  321GB free · ~500 MB/s read                       │
│  ┌─────────────────────────────────────────────┐    │
│  │ Cold expert pages (kernel auto-evict)       │    │
│  │ GGUF file mmap mapping                      │    │
│  └─────────────────────────────────────────────┘    │
│  Purpose: Cold storage, kernel LRU managed          │
└─────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Computation Offloading (from KTransformers)

MoE expert weights **stay in CPU RAM permanently** — they are NOT copied to GPU. The CPU executes expert forward passes directly. This avoids PCIe transfer bottlenecks (~32 GB/s PCIe vs ~440 GB/s DDR5 in server scenarios).

On consumer hardware (G4600 + GTX 1050), the same principle applies: experts in RAM, attention on GPU.

### 2. Arithmetic Intensity-Based Allocation

| Component | Arithmetic Intensity | Assigned To | Rationale |
|-----------|-------------------|-------------|-----------|
| Attention (MLA) | 512 (very high) | GPU VRAM | Compute-bound, benefits from GPU massively |
| MoE Routed Experts | ~0.075 (very low) | CPU RAM | Memory-bound, CPU large RAM is optimal |
| Shared Expert | Medium | GPU | Always active, fixed load |
| Embedding/Output | Low | GPU (via attention) | Small tensors, convenient |

### 3. mmap-Based Three-Tier Storage (from llama.cpp)

The GGUF model file is memory-mapped (mmap) into the process address space. The OS kernel's LRU page eviction algorithm naturally manages which pages live in RAM vs SSD:

- **Hot pages**: Frequently accessed expert weights stay in RAM
- **Cold pages**: Rarely used weights are automatically paged out to SSD
- **Zero copying**: No manual memory management needed

### 4. SSE Fallback (from llama.cpp)

llama.cpp uses runtime CPU dispatch: it compiles multiple code paths (SSE, AVX, AVX2, AVX512) and selects the best one at startup. This is critical for CPUs without AVX (like Pentium G4600).

## Deployment Configurations

### Config A: Baseline (All Experts on CPU)

```bash
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 18 \
  --cpu-moe \
  -c 2048 -t 4 -b 512 -ub 2048 \
  --host 0.0.0.0 --port 8111
```

- VRAM: 1,728 MiB (43%)
- Speed: 0.65 tok/s
- All 256×40=10,240 experts on CPU via mmap

### Config B: Partial Expert GPU Offload (Last 4 Layers)

```bash
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 20 \
  --n-cpu-moe 36 \
  -c 2048 -t 4 -b 512 -ub 2048 \
  --host 0.0.0.0 --port 8111
```

- VRAM: 3,574 MiB (88%)
- Speed: 0.74 tok/s (+14%)
- Layers 36-39 (4 layers × 256 experts = 1,024 experts) on GPU

### Config C: Aggressive GPU Offload (Last 5 Layers)

```bash
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 18 \
  --n-cpu-moe 35 \
  -c 2048 -t 4 -b 512 -ub 2048 \
  --host 0.0.0.0 --port 8111
```

- VRAM: 3,976 MiB (98.4%)
- Speed: 0.82 tok/s (+26%)
- Layers 35-39 (5 layers × 256 experts = 1,280 experts) on GPU

## Limitations

1. **Static placement**: `--n-cpu-moe` assigns entire layers to GPU, not individual experts
2. **No frequency awareness**: Hot/cold expert distinction isn't used at placement time
3. **VRAM waste**: GPU may store rarely-used experts while frequently-used ones stay on slow CPU
4. **No runtime adaptation**: Expert placement is fixed at load time

The next step — **dynamic expert caching** — addresses all four limitations.
